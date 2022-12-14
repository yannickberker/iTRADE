"""Test pylint settings."""
import filecmp
from pathlib import Path

from .run_all import collect_arguments, download_data
from .run_cnn import run_cnn


def test_reproducibility(tmp_path: Path) -> None:
    """Test that run_all yields reproducible results."""
    download_data()
    run_dirs = {}
    for run in ("0same", "2diff", "1same"):
        run_id = f"ReproTest-{run}"
        results_dir = tmp_path
        seed = 0 if run.endswith("same") else 1
        args = collect_arguments(
            # fmt: off
            # define data
            "--run-id", run_id,
            "--dataset", "INF_R_153_CE",
            "--pretraining", "imagenet",
            # define test
            "--results-dir", results_dir,
            "--seed", seed,
            "--fast-try",
        )
        run_dirs[run] = run_cnn(*args)

    files = {
        file.relative_to(run_dir)
        for run_dir in run_dirs.values()
        for file in run_dir.glob("**/*")
        if file.is_file()
        # .pb files include autogenerated function suffixes which differ when saving a
        # model multiple times from the same process. Fortunately, we are happy enough
        # if the weights files are equal. Alternatively, we might replace
        # run_cnn(*args) above with subprocess.run(["itrade-run-cnn", *map(str, args)]).
        and file.name != "saved_model.pb"
    }
    assert len(files) > 0

    (equal, different, error) = filecmp.cmpfiles(
        run_dirs["0same"], run_dirs["2diff"], files, shallow=False
    )
    assert len(equal) < len(files)
    assert len(different) > 0
    assert len(error) == 0

    (equal, different, error) = filecmp.cmpfiles(
        run_dirs["0same"], run_dirs["1same"], files, shallow=False
    )
    assert len(equal) == len(files)
    assert len(different) == 0
    assert len(error) == 0
