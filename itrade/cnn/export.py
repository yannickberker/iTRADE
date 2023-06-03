"""Export CNN data."""
from argparse import Namespace
from gzip import GzipFile
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import RocCurveDisplay, roc_curve

from ..util import images, layouts
from ..util import toolbox as tb
from ..util.layouts import VIABILITY_SCALE
from .datasets import PREDICT, TRAIN, VAL

# Keep DIR_TRAIN and DIR_VAL in sync with
# https://github.com/keras-team/keras/blob/v2.8.0/keras/callbacks.py#L2269-L2273
DIR_TRAIN = "train"
DIR_VAL = "validation"
DIR_PRED = "predict"
DIR_TEST = "test"

N_ROC_EDGES = 12
N_ROC_VERTICES = 3

N_PCA_COMPONENTS = 0.95
TSNE_PERPLEXITY = 10
RAND_STATE = 0


def data2pptx(
    args: Namespace, data: dict[str, Any], dt: str, subdir: str, filename: str
) -> None:
    """Store data set as pptx."""
    input_ori = data[dt].get("input_ori")
    if input_ori is None:
        return
    coords = data[dt]["coord"]
    labels = data[dt]["label"][:, 0] if "label" in data[dt] else []
    preds = data[dt]["prediction"][:, 0] if "prediction" in data[dt] else []

    folder = args.run_dir / subdir
    folder.mkdir(parents=True, exist_ok=True)

    print(f"Writing {folder / filename}.")
    images.images2pptx(input_ori, coords, folder / filename, labels=labels, preds=preds)


def before_training(args: Namespace, data: dict[str, Any]) -> None:
    """Store training/validation data as pptx."""
    for dt, subdir in ((TRAIN, DIR_TRAIN), (VAL, DIR_VAL)):
        data2pptx(args, data, dt, subdir, f"inputs_and_labels_{dt}.pptx")


def after_prediction(args: Namespace, data: dict[str, Any]) -> None:
    """Store data after application of the network."""
    for dt, subdir in ((TRAIN, DIR_TRAIN), (VAL, DIR_VAL), (PREDICT, DIR_TEST)):
        data2pptx(args, data, dt, subdir, f"inputs_and_predictions_{dt}.pptx")

    folder = args.run_dir / DIR_PRED
    folder.mkdir(parents=True, exist_ok=True)

    # Export plate-reader-like files
    predictions = data[PREDICT]["prediction"]

    wells = data["layout"]
    wells = wells.set_index(layouts.COORD_COLS)
    wells_files = wells[wells.HasFiles].drop(columns=wells.columns)
    wells_files["Readout"] = (predictions[:, 1] * VIABILITY_SCALE).round()
    wells = wells.join(wells_files)
    wells = wells.reset_index()

    for plate in set(wells.Plate):
        plate_wells = wells.loc[wells.Plate == plate, ["Row", "Column", "Readout"]]
        df = plate_wells.set_index(["Row", "Column"])
        df = df.rename_axis([None, None], axis=0)
        df = df.unstack().Readout
        filename = f"{args.dataset}_{plate}.txt"
        print(f"Writing {folder / filename}.")
        df.to_csv(folder / filename, sep="\t", na_rep="NA", float_format="%.0f")

    # Export features
    features = data[PREDICT]["features"]
    filename = f"{args.dataset}.features.txt.gz"
    print(f"Writing {folder / filename}.")
    np.savetxt(GzipFile(folder / filename, "w", mtime=0), features)


def tsne(args: Namespace, data: dict[str, Any]) -> None:
    """Export t-SNE embeddings."""

    def mi_df_plus_array(midf: pd.DataFrame, key: str, arr: np.ndarray) -> pd.DataFrame:
        """Concat NumPy array to Pandas MultiIndex DataFrame using Multi-/RangeIndex."""
        tb.ensure(isinstance(midf.columns, pd.MultiIndex))
        midf.columns = cast(pd.MultiIndex, midf.columns)
        tb.ensure(midf.columns.nlevels == 2)
        tb.ensure(key not in midf.columns.levels[0])
        indices = pd.MultiIndex.from_product([[key], range(arr.shape[-1])])
        arr_df = pd.DataFrame(arr, index=midf.index, columns=indices)
        return pd.concat([midf, arr_df], axis=1)

    df = data["layout"]
    df = df[df.HasFiles]
    df.columns = pd.MultiIndex.from_product((["Layout"], df.columns))
    df = mi_df_plus_array(df, "Features", data[PREDICT]["features"])

    # Remove all but the largest concentration for each drug series
    grouped_conc = df.Layout.groupby("Treatment").Concentration
    df[("Concentration", "Fraction")] = grouped_conc.transform(
        lambda x: x / x.clip(lower=1e-100).max()
    )
    df[("Concentration", "Step")] = np.log10(
        df.Concentration.Fraction.to_numpy(),
        out=np.zeros_like(df.Concentration.Fraction),
        where=df.Concentration.Fraction > 0,
    )
    inj_log10 = len(set(df.Concentration.Step)) <= len(set(df.Concentration.Fraction))
    tb.ensure(inj_log10, "Non-injective log10.")
    df = df.loc[df.Concentration.Step == 0, :]

    # Compute PCA components
    pca = PCA(random_state=RAND_STATE, n_components=N_PCA_COMPONENTS).fit(df.Features)
    components = pca.transform(df.Features)
    var_sum = pca.explained_variance_ratio_.sum()
    print(f"Number of PCA components: {pca.n_components_}")
    print(f"Sum of explained variance: {100 * var_sum:0.2f}")
    df = mi_df_plus_array(df, "PCA", components)

    # Compute t-SNE embedding
    tsne_ = TSNE(
        perplexity=1 if args.fast_try else TSNE_PERPLEXITY,
        learning_rate="auto",
        init="random",
        random_state=RAND_STATE,
    )
    embedding = tsne_.fit_transform(df.PCA)
    df = mi_df_plus_array(df, "tSNE", embedding)

    # Export data frame
    df[("Layout", "Plate")] = df.Layout.Plate.astype(str)
    df = df.drop([("Layout", "File_proj"), ("Layout", "File_midz")], axis="columns")
    df.columns = df.columns.map(lambda x: "_".join(map(str, x)).strip("_"))

    folder = args.run_dir / DIR_PRED
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"{args.dataset}.pca_tsne.feather"
    print(f"Writing {folder / filename}.")
    df.reset_index().to_feather(folder / filename)


def roc(args: Namespace, data: dict[str, Any]) -> None:
    """Export ROC curves."""
    # Return if not in classification task
    if len(data["out_shape"]) > 1:
        return

    for data_type, subdir in (
        (TRAIN, DIR_TRAIN),
        (VAL, DIR_VAL),
        (PREDICT, DIR_TEST),
    ):
        if "label" not in data[data_type]:
            continue

        y_true = np.argmax(data[data_type]["label"], axis=1)
        y_pred = data[data_type]["prediction"][:, 1]

        folder = args.run_dir / subdir
        folder.mkdir(parents=True, exist_ok=True)

        prev_backend = mpl.get_backend()

        with plt.ioff():
            for backend, ext in (("pdf", "pdf"), ("agg", "png")):
                mpl.use(backend)

                axes = RocCurveDisplay.from_predictions(y_true, y_pred).ax_
                axes.set_aspect("equal")
                axes.set_xlim(-0.1, 1.1)
                axes.set_ylim(-0.1, 1.1)

                fpr, tpr, thresholds = roc_curve(y_true, y_pred)

                thresholds = [f"{t:.2g}" for t in thresholds]
                thresholds[0] = "+∞"
                thresholds.append("-∞")

                n_vert = len(fpr)
                n_edge = n_vert - 1
                step_vert = np.ceil(n_vert / (N_ROC_VERTICES - 1)).astype(int)
                step_edge = np.ceil(n_edge / (N_ROC_EDGES - 1)).astype(int)

                text_kwargs: dict[str, Any] = {
                    "bbox": {
                        "boxstyle": "round, pad=0.3",
                        "fc": "yellow",
                        "alpha": 0.75,
                    },
                    "fontsize": "xx-small",
                }
                vert_kwargs: dict[str, Any] = {
                    "rotation": 45,
                }
                edge_kwargs: dict[str, Any] = {
                    "ha": "center",
                    "va": "center",
                }

                for i in set(range(0, n_vert, step_vert)) | {n_vert - 1}:
                    text = f"({thresholds[i]}, {thresholds[i + 1]})"
                    x = fpr[i]
                    y = tpr[i]
                    vert_kwargs["ha"] = "left" if (x, y) == (1, 1) else "right"
                    vert_kwargs["va"] = "top" if (x, y) == (0, 0) else "bottom"
                    axes.text(x, y, text, **text_kwargs, **vert_kwargs)

                for i in range(0, n_edge, step_edge):
                    text = thresholds[i + 1]
                    x = (fpr[i] + fpr[i + 1]) / 2
                    y = (tpr[i] + tpr[i + 1]) / 2
                    axes.text(x, y, text, **text_kwargs, **edge_kwargs)

                filename = folder / f"roc_and_auc_{data_type}.{ext}"
                print(f"Writing {filename}.")
                axes.figure.savefig(filename, bbox_inches="tight")

        mpl.use(prev_backend, force=False)
