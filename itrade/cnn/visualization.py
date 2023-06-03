"""Visualize CNN data."""
import warnings
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, TypeAlias

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

try:
    from tf_keras_vis.activation_maximization import ActivationMaximization
    from tf_keras_vis.activation_maximization.callbacks import Progress
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.saliency import Saliency
    from tf_keras_vis.scorecam import ScoreCAM

    USE_KERAS_VIS = True
except ImportError:
    USE_KERAS_VIS = False

from ..util import toolbox as tb
from ..util.images import KERAS_MAX, gray2rgb

Model: TypeAlias = keras.models.Model


VIS_METHODS = ("GradCAM", "GradCAM++", "ScoreCAM")


def vis_model(args: Namespace, data: dict[str, Any], model: Model) -> None:
    """Visualize class-activation maps."""
    if not USE_KERAS_VIS:
        warnings.warn("Skipping model visualization due to tf_keras_vis import error.")
        return

    wells = data["layout"]
    wells = wells[wells.HasFiles]
    images_input = data["predict"]["input"]
    images_proj = data["predict"]["images"][:, :, :, 0]
    predictions = data["predict"]["prediction"]
    tb.ensure(len(wells) == len(images_input) == len(images_proj))

    wells = wells.reset_index(drop=True).rename_axis("i").reset_index()

    if args.dataset == "INF_R_1025_primary_V2_DS1":

        def first_wells_per_plate(well_type: str) -> pd.DataFrame:
            return (
                wells[wells.WellType == well_type]
                .sort_values(["Column", "Row"])
                .groupby("Plate")
                .first()
            )

        wells_of_interest = pd.concat(
            [
                first_wells_per_plate("neg"),
                first_wells_per_plate("pos"),
                first_wells_per_plate("BzCl"),
                wells[(wells.Treatment == "Foretinib") & (wells.Column > 12)],
            ]
        )
    else:
        warnings.warn(f"Skipping model visualization for dataset '{args.dataset}'.")
        return

    classes = data["classes"]

    for index, well in wells_of_interest.set_index("i").iterrows():
        image_input = images_input[[index]]
        prediction = predictions[index]

        tb.ensure(np.isclose(model.predict(image_input), prediction).all())

        image_bg = gray2rgb(images_proj[index], scale=5) / KERAS_MAX
        fname_pattern = (
            f"{args.dataset}_{well.Treatment}_{well.CoordString}_{{suffix}}.png"
        )

        folder = args.run_dir / "visualization"
        folder.mkdir(parents=True, exist_ok=True)
        filename = Path(fname_pattern.format(suffix=f"[{prediction[1]:.2f}]"))
        plt.imsave(folder / filename, image_bg)

        for method in VIS_METHODS:
            save_keras_vis(
                model,
                method,
                image_input,
                classes,
                prediction,
                folder,
                fname_pattern,
                image_bg,
            )


def save_keras_vis(
    model: Model,
    method: str,
    image_input: np.ndarray,
    classes: Iterable[str],
    prediction: float,
    folder: Path,
    fname_pattern: str,
    image_bg: np.ndarray | None = None,
) -> None:
    """Visualize Keras model using tf-keras-vis."""
    # tf-keras-vis>=0.6.0 normalizes maps to [0, 1], so one-sided cmaps are appropriate
    vis_cmap = "Reds"

    # Optionally, keep softmax to make DMSO GradCAM weights the negative of STS weights
    def linear_activation_model(model: Model, remove_softmax: bool = True) -> Model:
        # Replace final softmax by linear activation
        tb.ensure(model.layers[-1].activation is keras.activations.softmax)
        if remove_softmax:
            model.layers[-1].activation = keras.activations.linear
        return model

    def last_conv_model(model: Model) -> Model:
        last_conv_layer = next(
            layer
            for layer in reversed(model.layers)
            if isinstance(layer, keras.layers.Conv2D)
        )
        model = keras.Model(inputs=model.inputs, outputs=last_conv_layer.output)
        model.layers[-1].activation = keras.activations.linear
        return model

    def get_class_score(
        output: tf.Tensor, cam_class: int, compare_scores: bool
    ) -> tf.Tensor:
        if compare_scores:
            # compare computed outputs to external predictions: equality comparison
            # fails with ScoreCAM due to masked inputs (but scores can also be better),
            # and with SmoothGrad due to added noise (scores are usually worse)
            print("prediction :", prediction)
            full_output = (
                output
                if np.isclose(sum(output[0]), 1.0, atol=1e-6)
                else tf.nn.softmax(output)
            )
            print("full output:", np.array(full_output))

        # ":" to handle multiple inputs
        outputs = output[:, cam_class]
        return outputs

    def get_filter_score(
        output: tf.Tensor, cam_filter: int, top_row: bool
    ) -> tf.Tensor:
        # first ":" to handle multiple inputs
        if top_row:
            outputs = output[:, cam_filter, :, :]
        else:
            outputs = output[:, :, :, cam_filter]
        return outputs

    model_modifier: Callable[[Any], Any] = linear_activation_model
    vis_args = [image_input]
    vis_kwargs: dict[str, Any] = {}
    compare_scores = True
    if method == "Saliency":
        ModelVisualization = Saliency
    elif method == "SmoothGrad":
        ModelVisualization = Saliency
        vis_kwargs = {"smooth_samples": 20}
    elif method == "GradCAM":
        ModelVisualization = Gradcam
    elif method == "GradCAM++":
        ModelVisualization = GradcamPlusPlus
    elif method == "ScoreCAM":
        ModelVisualization = ScoreCAM
        model_modifier = partial(linear_activation_model, remove_softmax=False)
    elif method == "Faster-ScoreCAM":
        ModelVisualization = ScoreCAM
        model_modifier = partial(linear_activation_model, remove_softmax=False)
        vis_kwargs = {"max_N": 1}
    elif method in ("Dense", "Conv", "ConvTop"):
        ModelVisualization = ActivationMaximization
        if method.startswith("Conv"):
            model_modifier = last_conv_model
        vis_args = []
        tf.random.set_seed(0)
        seed_input = tf.random.uniform((1, *model.input.shape[1:]), 0, KERAS_MAX)
        vis_kwargs = {
            "seed_input": seed_input,
            "callbacks": [Progress()],
        }
        compare_scores = False
    else:
        tb.ensure(False, f"Unknown method {method}")

    model_vis = ModelVisualization(model, model_modifier, clone=True)

    for cam_class, class_name in enumerate(classes):
        if method == "Conv":
            get_score = partial(
                get_filter_score,
                cam_filter=cam_class,
            )
        elif method == "ConvTop":
            get_score = partial(
                get_filter_score,
                cam_filter=cam_class,
                top_row=True,
            )
        else:
            get_score = partial(
                get_class_score,
                cam_class=cam_class,
                compare_scores=compare_scores,
            )

        vis_maps = model_vis(get_score, *vis_args, **vis_kwargs)
        vis_map = vis_maps[0]

        if method in ("Dense", "Conv", "ConvTop"):
            result = vis_map.astype(np.uint8)
        else:
            if image_bg is None:
                image_bg = np.zeros((*image_input.shape[:-1], 3))

            map_rgb = mpl.colormaps[vis_cmap](vis_map)[:, :, :3]

            alpha = np.clip(vis_map, 0, 0.5)
            alpha = np.expand_dims(alpha, -1)

            result = image_bg * (1 - alpha) + map_rgb * alpha

            result = result.astype(np.float32)

        filename = Path(fname_pattern.format(suffix=f"{method}_{class_name}"))
        plt.imsave(folder / filename, result)
