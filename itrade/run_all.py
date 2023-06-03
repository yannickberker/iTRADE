"""Run a sequence of CNN trainings reproducing paper results."""
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from tensorflow import keras

from . import (
    BASE_DIR,
    DATA_DIR,
    DATA_ZIP_NAME,
    DIR_ENV_VAR,
    ITREX_DIR,
    PACKAGE_NAME,
    PACKAGE_SHORT,
    PLOTS_DIR,
    RESULTS_DIR,
)
from .cnn.export import DIR_PRED
from .run_cnn import run_cnn
from .util import itrex
from .util import toolbox as tb

DATA_ZIP_URL = "https://zenodo.org/record/5885481/files/" + DATA_ZIP_NAME
DATA_ZIP_HASH = "8e829e96807d4d789817899ba704a87c"

CE_SUFFIX = "_CE"

BATCH_SIZE = 1
EPOCHS = 200
LABEL_SMOOTHING = 0.4
LEARNING_RATE_PHASE23 = 3e-07

MODEL_FILE_NAME = "model.tf"


def collect_arguments(*args: Any) -> tuple[Any, ...]:
    """Return standard parameters."""
    return (
        # fmt: off
        "--data-dir", DATA_DIR,
        "--mixed-precision",
        "--model", "VGG19",
        "--shift-images",
        "--label-smoothing", LABEL_SMOOTHING,
        "--batch-size", BATCH_SIZE,
        "--metric", "categorical_accuracy",
        "--epochs", EPOCHS,
        "--verbosity", 2,
        "--results-dir", RESULTS_DIR,
        "--export-data",
        "--save-model",
        "--predict-model",
        "--test-export-tsne",
        # use new GPU allocator for repeated training to reduce the rate of OOM issues
        "--use-cuda-malloc-async",
        *(sys.argv[1:] if not os.getenv("PYTEST_CURRENT_TEST") else []),
        *args
    )


def download_data() -> None:
    """Download and extract data if required."""
    if DATA_DIR.is_dir() and next(DATA_DIR.glob("*"), None) is not None:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.get_file(
        fname=DATA_ZIP_NAME,
        origin=DATA_ZIP_URL,
        file_hash=DATA_ZIP_HASH,
        extract=True,
        cache_dir=DATA_DIR.parent,
        cache_subdir=DATA_DIR.name,
    )


def run_all() -> None:
    """Run CNN training, and process and visualize results."""
    run_cnns()
    run_itrex()
    run_plots()


def run_cnns() -> None:
    """Run a sequence of CNN trainings reproducing paper results."""
    print("Hello there!")
    print(f"- Using {BASE_DIR} as base dir, set the {DIR_ENV_VAR} env var to override.")
    print(f"- Working with data in {DATA_DIR}.")
    print(f"- Storing results in {RESULTS_DIR}.")
    print("- Continuing in 5 seconds...")
    time.sleep(5)

    download_data()

    # Read available screen IDs
    sids = sorted(f.name for f in DATA_DIR.glob("Images/*") if f.is_dir())

    control_sids = (s for s in sids if s.endswith(CE_SUFFIX))
    control_sid = next(control_sids)
    tb.ensure(not next(control_sids, False), "More than one control experiment found")

    screen_sids = [s for s in sids if s != control_sid]

    common_args = collect_arguments("--tb-log-dir", RESULTS_DIR / "Board")

    # Run Phase-1 pretraining
    phase1_dir = run_cnn(
        *common_args,
        # fmt: off
        "--run-id", f"Phase1-{control_sid}",
        "--dataset", control_sid,
        "--pretraining", "imagenet",
        "--checkpoint-model-file-name", "model_checkpoint_e{epoch:03d}_b{batch:03d}.tf",
        "--checkpoint-period", 20,
        "--val-model",
        "--val-samples", 20,
        "--metric", "AUC",
        "--test-export-roc",
    )

    # Run Phase-2/-3 training for each sample
    phase2_dirs: list[Path] = []
    phase3_dirs: list[Path] = []
    for sid in screen_sids:
        # Phase 2
        phase2_dir = run_cnn(
            *common_args,
            # fmt: off
            "--run-id", f"Phase2-{sid}",
            "--dataset", sid,
            "--load-model",
            "--load-model-file-name", phase1_dir / MODEL_FILE_NAME,
            "--val-model",
            "--val-samples", 6,
            "--learning-rate", LEARNING_RATE_PHASE23,
            "--metric", "AUC",
            "--vis-model",
            "--test-export-roc",
        )
        phase2_dirs.append(phase2_dir)

        # Phase 3
        phase3_dir = run_cnn(
            *common_args,
            # fmt: off
            "--run-id", f"Phase3-{sid}",
            "--dataset", sid,
            "--load-model",
            "--load-model-file-name", phase2_dir / MODEL_FILE_NAME,
            "--ct-files", phase2_dir / DIR_PRED / f"{sid}_{{P}}.txt",
            "--learning-rate", LEARNING_RATE_PHASE23,
            "--vis-model",
        )
        phase3_dirs.append(phase3_dir)

    # Quantify the effect of transfer learning ...
    control_sample = control_sid.removesuffix(CE_SUFFIX)
    transfer_sid = next(s for s in screen_sids if s.startswith(control_sample))

    # ... in general (by skipping pretraining and Phase 1)
    run_cnn(
        *common_args,
        # fmt: off
        "--pretraining", "None",
        "--run-id", f"TransferStudy-Phase2-{transfer_sid}-NoPretraining",
        "--dataset", transfer_sid,
        "--val-model",
        "--val-samples", 6,
        "--metric", "AUC",
        "--test-export-roc",
    )

    # ... from Phase 1 to Phase 2 (by skipping Phase 1)
    run_cnn(
        *common_args,
        # fmt: off
        "--run-id", f"TransferStudy-Phase2-{transfer_sid}-OnlyImageNet",
        "--dataset", transfer_sid,
        "--pretraining", "imagenet",
        "--val-model",
        "--val-samples", 6,
        "--metric", "AUC",
        "--test-export-roc",
    )

    # ... from Phase 1 to Phase 2 (by starting Phase 2 from early Phase-1 checkpoints)
    for model_file_name in phase1_dir.glob("model_checkpoint_e*.tf"):
        match = re.search(
            r"(?<=^model_checkpoint_e)\d+(?=_.*.tf$)", model_file_name.name
        )
        if match is None:
            raise ValueError("Unknown model file name")
        run_cnn(
            *common_args,
            # fmt: off
            "--run-id", f"TransferStudy-Phase2-{transfer_sid}-Phase1Epoch-{match[0]}",
            "--dataset", transfer_sid,
            "--load-model",
            "--load-model-file-name", model_file_name,
            "--val-model",
            "--val-samples", 6,
            "--metric", "AUC",
            "--test-export-roc",
        )

    # Not shown in manuscript: try other model structures (Phase 1 and 2)
    # using the exact same settings as above.
    structures = ("DenseNet121", "MobileNetV2", "ResNet50", "ResNet50V2", "VGG16")
    # Phase1: "VGG16" slightly worse (but exactly the same as when using "VGG16" in
    # `collect_arguments`); all others do not generalize.

    # Also tried ("NoShift"):
    # > common_args = tuple(arg for arg in common_args if arg != "--shift-images")
    # Phase1: VGG16 replicates training results almost exactly, but validation is poor.
    # (This may have to do with data augmentation, in particular, `brightness_range`.)
    # Performance of all others remains as poor as with `--shift-images`.

    # Also tried ("NoShiftNoDa"):
    # > common_args = tuple(arg for arg in common_args if arg != "--shift-images")
    # > common_args += ("--skip-da",)
    # Phase1: VGG16 trains and validates well, very fast.
    # (It remains to be seen how well that generalizes to Phase2.)
    # Others train better, but still do not generalize.

    for structure in structures:
        struc_phase1_dir = run_cnn(
            *common_args,
            # fmt: off
            "--model", structure,
            "--run-id", f"StructureStudy-Phase1-{control_sid}-{structure}",
            "--dataset", control_sid,
            "--pretraining", "imagenet",
            "--val-model",
            "--val-samples", 20,
            "--metric", "AUC",
            "--test-export-roc",
        )

        struc_phase2_dir = run_cnn(
            *common_args,
            # fmt: off
            "--model", structure,
            "--run-id", f"StructureStudy-Phase2-{transfer_sid}-{structure}",
            "--dataset", transfer_sid,
            "--load-model",
            "--load-model-file-name", struc_phase1_dir / MODEL_FILE_NAME,
            "--val-model",
            "--val-samples", 6,
            "--learning-rate", LEARNING_RATE_PHASE23,
            "--metric", "AUC",
            "--test-export-roc",
        )

        run_cnn(
            *common_args,
            # fmt: off
            "--run-id", f"StructureStudy-Phase3-{transfer_sid}-{structure}",
            "--dataset", transfer_sid,
            "--load-model",
            "--load-model-file-name", struc_phase2_dir / MODEL_FILE_NAME,
            "--ct-files", struc_phase2_dir / DIR_PRED / f"{transfer_sid}_{{P}}.txt",
            "--learning-rate", LEARNING_RATE_PHASE23,
        )

    print("Done.")


def run_itrex() -> None:
    """Collect data, process in iTReX, and store to disk."""
    ITREX_DIR.mkdir(exist_ok=True)
    met_layout_file = itrex.download_demo_file(itrex.MET_LAYOUT_FILE, ITREX_DIR)
    ima_layout_file = itrex.download_demo_file(itrex.IMA_LAYOUT_FILE, ITREX_DIR)

    plate_suffix_pattern = r"(?<=_)P(.)(_H104-03N\1[\w-]\d{2})?(?=\.txt$)"

    exception = None
    for folder, long_name, move_prefix, short_name, layout_file in (
        (DATA_DIR / "Metabolic", "Metabolic", "", "Met", met_layout_file),
        (DATA_DIR / "MeanOfStack", "MeanOfStack", "", "MoS", ima_layout_file),
        (RESULTS_DIR, PACKAGE_NAME, "Phase2", f"{PACKAGE_SHORT}2", ima_layout_file),
        (RESULTS_DIR, PACKAGE_NAME, "Phase3", f"{PACKAGE_SHORT}3", ima_layout_file),
    ):
        if move_prefix:
            long_name += "-" + move_prefix
        sid_dirs = folder.glob(f"{move_prefix}*")
        readouts_file = ITREX_DIR / f"iTReX-Input-{long_name}.zip"
        with ZipFile(readouts_file, "w") as zip_file:
            for sid_dir in sorted(sid_dirs):
                sid_dir_name = sid_dir.name
                if move_prefix:
                    sid_dir_name = re.sub(f"^{move_prefix}-", "", sid_dir_name)
                arc_folder = Path(short_name + "-" + sid_dir_name)
                if (sid_dir / "predict").is_dir():
                    sid_dir /= "predict"
                for file in sorted(sid_dir.glob("*.txt")):
                    base_name = re.sub(plate_suffix_pattern, r"\1", file.name)
                    arc_name = arc_folder / (short_name + "-" + base_name)
                    zip_file.write(file, arc_name)

        try:
            itrex.process_cohort(layout_file, readouts_file, ITREX_DIR, name=long_name)
        except Exception as ex:  # pylint: disable=broad-except
            exception = exception or ex
            warnings.warn(f"""
                Automated iTReX cohort processing failed. Feel free to try it manually:
                - Visit {itrex.ITREX_URL} in your browser
                - Accept terms and conditions
                - For "Number of Samples", select "Cohort"
                - As "Layout Table", upload {layout_file}
                - As "Readout Matrices", upload {readouts_file}
                - Click "Start One-Click Analysis"
                - After processing has finished, click "Download Results"
                - Copy or move the downloaded file to {ITREX_DIR}
                """)
    if exception is not None:
        raise exception


def run_plots() -> None:
    """Generate plots using R via rpy2."""
    try:
        # pylint:disable=import-outside-toplevel
        from rpy2 import robjects
    except ModuleNotFoundError:
        print("""
            Cannot import `rpy2` module to generate plots."
            Try installing `R` and reinstalling iTRADE with the `R` extra.
            If `rpy2` continues to fail, try `Rscript itrade/util/plots.R`.
            """)
        return

    script_file = Path(__file__).parent / "util" / "plots.R"
    robjects.r.source(str(script_file))
    dirs = (DATA_DIR, RESULTS_DIR, ITREX_DIR, PLOTS_DIR)
    robjects.r.plots(*map(str, dirs))


if __name__ == "__main__":
    run_all()
