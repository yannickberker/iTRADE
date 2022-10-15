"""Loading networks models for CNN."""
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Any, Callable

from tensorflow import keras

from ..util import toolbox as tb
from . import keras_app


def add_arguments(parser: ArgumentParser) -> None:
    """Add model arguments."""
    model = parser.add_argument_group("model")
    model.add_argument(
        "--model",
        type=str,
        choices=list(keras_app.MODELS),
        help="network model (if not using --load-model)",
    )
    model.add_argument(
        "--pretraining",
        type=lambda weights: None if weights == "None" else weights,
        choices=[None, "imagenet"],
        help="pretrained weights (if not using --load-model)",
    )


def create_model(args: Namespace, data: dict[str, Any]) -> keras.models.Model:
    """Create a model based on the 'model' argument."""
    tb.ensure(args.model in keras_app.MODELS, "Unknown model.")

    in_shape = data["in_shape"]
    n_classes = data["out_shape"][-1]
    model = keras_app.keras_app_model(
        args.model,
        weights=args.pretraining,
        input_shape=in_shape,
        n_classes=n_classes,
    )

    model.summary(line_length=120)
    return model


def get_activation(act: str) -> Callable:
    """Convert activation string/function into activation function."""
    return keras.activations.get(act)


def configure_loss(loss: str, label_smoothing: float = 0) -> keras.losses.Loss:
    """Generate a loss function with per-sample, per-pixel weights."""
    loss_fcn = keras.losses.get(loss)
    if label_smoothing > 0:
        name = loss_fcn.__name__
        loss_fcn = partial(loss_fcn, label_smoothing=label_smoothing)
        loss_fcn.__name__ = name
    return loss_fcn


def load_model(args: Namespace, data: dict[str, Any]) -> keras.models.Model:
    """Load model from saved file using shape from data."""
    loss_fcn = configure_loss(args.loss, args.label_smoothing)
    shape = data["in_shape"][1:]

    path = args.run_dir / args.load_model_file_name
    model = keras.models.load_model(path, compile=False)
    model.compile(loss=loss_fcn)
    model.build(shape)
    return model


def save_model(args: Namespace, model: keras.models.Model) -> None:
    """Save the model to the file system."""
    path = args.run_dir / args.save_model_file_name
    model.save(path)
