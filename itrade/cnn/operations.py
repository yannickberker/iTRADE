"""Implement CNN operations: training and testing."""
import math
import shlex
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable, TypeAlias

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras

from ..util import toolbox as tb
from .datasets import DATA_TYPES, TRAIN, VAL
from .export import DIR_TRAIN, DIR_VAL
from .models import configure_loss

Layer: TypeAlias = keras.layers.Layer


def add_arguments(parser: ArgumentParser) -> None:
    """Add operation arguments."""
    training = parser.add_argument_group("training")
    training.add_argument(
        "--loss",
        "--loss-fcn",
        default="categorical_crossentropy",
        type=str,
        help="loss function",
    )
    training.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help=r"use label smoothing, \in [0, 1]",
    )
    training.add_argument(
        "--metric",
        action="append",
        type=str,
        help="metric to compute during training (can be used multiple times)",
        dest="metrics",
    )
    training.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        help="optimizer (e.g., adam, sgd, ...)",
    )
    training.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-07,
        help="learning rate for optimizer",
    )
    training.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=1,
        help="number of epochs",
    )
    training.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="batch size",
    )
    training.add_argument(
        "--red-lr",
        action="store_true",
        help="learning rate reduction",
    )
    training.add_argument(
        "--red-lr-parms",
        nargs=4,
        default=[2, 0.2, 3, 1e-10],
        type=float,
        help="patience, factor, cooldown, min_lr",
    )
    training.add_argument(
        "--stop-early-epochs",
        type=int,
        help="stop training early based on val_loss after than many epochs ",
    )
    training.add_argument(
        "--checkpoint-model-file-name",
        type=Path,
        default="model_checkpoint.tf",
        help="output model checkpoints to this file",
    )
    training.add_argument(
        "--checkpoint-period",
        type=int,
        default=0,
        help="how often to save checkpoints (in epochs)",
    )

    logging = parser.add_argument_group("logging")
    logging.add_argument(
        "--tb-log-dir",
        type=Path,
        help="log directory for TensorBoard output (creates subdirectory)",
    )
    logging.add_argument(
        "--tb-write-images",
        action="store_true",
        help="write input images to TensorBoard for validation of data augmentation",
    )
    logging.add_argument(
        "--verbosity",
        "-v",
        type=int,
        default=0,
        help="verbosity",
    )


def train_model(
    args: Namespace, data: dict[str, Any], model: keras.models.Model
) -> keras.models.Model:
    """Run actual training."""
    # Optimizer
    optimizer = keras.optimizers.get(args.optimizer)
    optimizer.lr = args.lr

    # Loss
    loss_fcn = configure_loss(args.loss, args.label_smoothing)

    # Callbacks

    all_callbacks = []

    # All TensorBoard

    if args.tb_log_dir:

        # TensorBoard logs

        # Use unique directories to separate repeated runs:
        tb_run_id = args.run_id + time.strftime("_%Y-%m-%d_%H-%M-%S")
        tb_run_dir = args.tb_log_dir / tb_run_id
        # Ideally, we would also use unique subdirs for all callback writers due to
        # https://github.com/tensorflow/tensorboard/issues/1011
        # However, this increases the number of "runs" shown.
        # Fortunately, there is the (experimental) command-line parameter
        #   --reload_multifile=true --reload_multifile_inactive_secs 3600
        # which one can use to see all files appearing live during training
        # ("or to just restart TensorBoard when you want to see the full data").
        tb_train_dir = tb_run_dir / DIR_TRAIN
        tb_val_dir = tb_run_dir / DIR_VAL

        tb_callback = keras.callbacks.TensorBoard(log_dir=tb_run_dir, histogram_freq=9)
        all_callbacks.append(tb_callback)

        # TensorBoard activation histograms

        class ActivationHistogramCallback(keras.callbacks.Callback):
            """Output activation histograms."""

            def __init__(self, layers: list[Layer]):
                self.layers = layers
                self.inner_calls: dict[Layer, Callable] = {}
                self.outer_calls: dict[Layer, Callable] = {}
                self.batch_outputs: dict[Layer, tf.Variable] = {}
                self.mode_tbdirs = {TRAIN: tb_train_dir, VAL: tb_val_dir}
                self.epoch_outputs: dict[str, dict[Layer, list[tf.Variable]]] = {
                    mode: {} for mode in self.mode_tbdirs
                }
                self.writers = {
                    mode: tf.summary.create_file_writer(str(tb_dir))
                    for mode, tb_dir in self.mode_tbdirs.items()
                }

            def set_model(self, _model: keras.models.Model) -> None:
                """Prepare new model layer calls when model is set."""
                for layer in self.layers:
                    self.batch_outputs[layer] = tf.Variable(
                        math.nan, dtype=layer.output.dtype, shape=tf.TensorShape(None)
                    )

                    self.inner_calls[layer] = layer.call

                    @tf.function
                    def outer_call(
                        inputs: tf.Tensor,
                        *args: Any,
                        _ahc_layer: Layer = layer,
                        **kwargs: Any,
                    ) -> tf.Tensor:
                        outputs = self.inner_calls[_ahc_layer](inputs, *args, **kwargs)
                        self.batch_outputs[_ahc_layer].assign(outputs)
                        return outputs

                    self.outer_calls[layer] = outer_call

            def on_epoch_begin(
                self, _epoch: int, _logs: dict[str, float] = None
            ) -> None:
                """Reset epoch outputs."""
                for mode in self.mode_tbdirs:
                    for layer in self.layers:
                        self.epoch_outputs[mode][layer] = []

            def on_epoch_end(self, epoch: int, _logs: dict[str, float] = None) -> None:
                """Write epoch output histograms."""
                for mode in self.mode_tbdirs:
                    with self.writers[mode].as_default():
                        for layer, data in self.epoch_outputs[mode].items():
                            if isinstance(layer, keras.layers.InputLayer):
                                continue
                            name = f"{layer.name}/output"
                            tf.summary.histogram(name, data, step=epoch)

            def on_train_batch_begin(
                self, _batch: int, _logs: dict[str, float] = None
            ) -> None:
                """Wrap layer calls for this batch."""
                self.wrap_layer_calls()

            def on_train_batch_end(
                self, _batch: int, _logs: dict[str, float] = None
            ) -> None:
                """Restore layer calls and write training batch histograms."""
                self.restore_layer_calls()
                self.append_batch_to_epoch_data(TRAIN)

            def on_test_batch_begin(
                self, _batch: int, _logs: dict[str, float] = None
            ) -> None:
                """Wrap layer calls for this batch."""
                self.wrap_layer_calls()

            def on_test_batch_end(
                self, _batch: int, _logs: dict[str, float] = None
            ) -> None:
                """Restore layer calls and write validation batch histograms."""
                self.restore_layer_calls()
                self.append_batch_to_epoch_data(VAL)

            def wrap_layer_calls(self) -> None:
                """Replace the calls during each batch."""
                for layer, outer_call in self.outer_calls.items():
                    layer.call = outer_call

            def restore_layer_calls(self) -> None:
                """Restore original calls after each batch (for ModelCheckpoints)."""
                for layer, inner_call in self.inner_calls.items():
                    layer.call = inner_call

            def append_batch_to_epoch_data(self, mode: str) -> None:
                """Append current batch data to epoch train/val data."""
                for layer, data in self.batch_outputs.items():
                    self.epoch_outputs[mode][layer].append(tf.Variable(data))

        # (histogram_freq, layers) and (epoch, step) times for VGG16/Phase1-INF_R_153_CE
        # 0     []      10s/epoch - 51ms/step
        # 1     []      19s/epoch - 98ms/step --> too much, choose histogram_freq ~= 10
        # 0     [-1:]   10s/epoch - 51ms/step --> ~ instantaneous for final layer
        # 1     [-1:]   18s/epoch - 97ms/step --> same conclusions as above
        # 0     [-3:]   11s/epoch - 56ms/step --> still very reasonable
        # 0     [-5:]   11s/epoch - 58ms/step --> still fast (but OOM on laptop)
        # 0     [:]                           --> OOM
        ah_callback = ActivationHistogramCallback(model.layers[-5:])
        all_callbacks.append(ah_callback)

        # TensorBoard HParams

        def keep_hparam(key: str, element: Any) -> bool:
            return not (key == "run_id" or isinstance(element, Path))

        def to_hparam(element: Any) -> bool | float | int | str:
            if isinstance(element, (tuple, list)):
                hparams = type(element)(map(to_hparam, element))
                return str(hparams)

            if isinstance(element, (bool, float, int, str)):
                return element

            if element is None or isinstance(element, dict):
                return str(element)

            if hasattr(element, "__name__"):
                return element.__name__

            return type(element).__name__

        hparams = {
            key: to_hparam(element)
            for (key, element) in vars(args).items()
            if keep_hparam(key, element)
        }
        hp_callback = hp.KerasCallback(str(tb_train_dir), hparams, trial_id=args.run_id)
        all_callbacks.append(hp_callback)

        # TensorBoard command line

        cmd_writer = tf.summary.create_file_writer(str(tb_train_dir))

        def log_command_line(_logs: dict[str, float] = None) -> None:
            """Write command line."""
            cmd = " ".join(map(shlex.quote, args.args_str))
            with cmd_writer.as_default():
                tf.summary.text("Command line", cmd, step=0)

        cmd_callback = keras.callbacks.LambdaCallback(on_train_begin=log_command_line)
        all_callbacks.append(cmd_callback)

        # TensorBoard batch images

        if args.tb_write_images:

            class InputImageCallback(keras.callbacks.Callback):
                """Write individual batch images to TensorBoard."""

                def __init__(self) -> None:
                    self.writer = tf.summary.create_file_writer(str(tb_train_dir))
                    self.epoch: int = -1
                    self.image: tf.Variable = None

                def set_model(self, model: keras.models.Model) -> None:
                    """Wrap the model.train_step function to access training images."""
                    self.image = tf.Variable(
                        math.nan, dtype=model.input.dtype, shape=tf.TensorShape(None)
                    )

                    model_train_step = model.train_step

                    def outer_train_step(
                        data: tuple[tf.Tensor, ...]
                    ) -> dict[str, tf.Tensor]:
                        # https://github.com/keras-team/keras/...
                        # .../blob/v2.8.0/keras/engine/training.py#L856
                        x, _y, _w = keras.utils.unpack_x_y_sample_weight(data)
                        self.image.assign(x)

                        return model_train_step(data)

                    model.train_step = outer_train_step

                def on_epoch_begin(
                    self, epoch: int, _logs: dict[str, float] = None
                ) -> None:
                    """Store epoch to use in export."""
                    self.epoch = epoch

                def on_train_batch_end(
                    self, batch: int, _logs: dict[str, float] = None
                ) -> None:
                    """Output the variables as images."""
                    name = f"input_e{1 + self.epoch}"

                    image_min = tf.reduce_min(self.image, axis=None)
                    image_max = tf.reduce_max(self.image, axis=None)
                    data = (self.image - image_min) / (image_max - image_min)

                    with self.writer.as_default():
                        tf.summary.image(name, data, step=batch)

            im_callback = InputImageCallback()
            all_callbacks.append(im_callback)

    # Model checkpoints

    if args.checkpoint_period > 0:
        name_is_fixed = "{" not in args.checkpoint_model_file_name.name

        # If the checkpoint model name is variable, save all models regardless of loss;
        # otherwise, save only the best-performing one (if possible - needs VAL data).
        save_best_only = name_is_fixed and len(data[VAL].get("input", ())) > 0

        cp_callback = keras.callbacks.ModelCheckpoint(
            args.run_dir / args.checkpoint_model_file_name,
            save_best_only=save_best_only,
            monitor="val_loss",
            verbose=args.verbosity,
            save_freq=args.checkpoint_period * data["fit"]["steps_per_epoch"],
        )
        all_callbacks.append(cp_callback)

    # Early stopping

    if args.stop_early_epochs:
        es_callback = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            verbose=1,
            patience=args.stop_early_epochs,
            restore_best_weights=True,
        )
        all_callbacks.append(es_callback)

    # Learning rate reduction

    if args.red_lr:
        parm = args.red_lr_parms
        red_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            verbose=1,
            patience=parm[0],
            factor=parm[1],
            cooldown=parm[2],
            min_lr=parm[3],
        )
        all_callbacks.append(red_lr_callback)

    # Compilation
    metrics = [keras.metrics.get(m) for m in args.metrics]

    model.compile(optimizer=optimizer, loss=loss_fcn, metrics=metrics)

    # Training
    start = tb.now()
    history = model.fit(
        **data["fit"],
        epochs=2 if args.fast_try else args.epochs,
        callbacks=all_callbacks,
        verbose=args.verbosity,
    )
    tb.print_elapsed(start, "Fitting", history.epoch[-1] + 1, "epoch")

    return model


def predict_model(
    args: Namespace, data: dict[str, Any], model: keras.models.Model
) -> dict[str, Any]:
    """Use trained network to make a prediction."""
    feature_layer = next(
        layer for layer in reversed(model.layers) if layer.output.shape[-1] > 2
    )
    output_vars = [model.output, feature_layer.output]
    vars_model = keras.Model(inputs=model.inputs, outputs=output_vars)

    for data_type in DATA_TYPES:
        inputs = data[data_type].get("input")
        if inputs is None:
            continue

        n_inputs = inputs.shape[0]

        start = tb.now()
        outputs = vars_model.predict(
            inputs, batch_size=args.batch_size, verbose=args.verbosity
        )
        tb.print_elapsed(start, f"Model application ({data_type})", n_inputs, "sample")

        data[data_type]["prediction"] = outputs[0]
        data[data_type]["features"] = outputs[1]

    return data
