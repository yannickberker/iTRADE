"""Configure random seeds and TensorFlow."""
import os
from argparse import ArgumentParser, Namespace

import tensorflow as tf
from tensorflow import keras


def add_arguments(parser: ArgumentParser) -> None:
    """Add configuration arguments."""
    tensorflow = parser.add_argument_group("tensorflow")
    tensorflow.add_argument(
        "--log-level",
        type=int,
        default=1,
        help="log level for C++ part of TensorFlow",
    )
    tensorflow.add_argument(
        "--mixed-precision",
        action="store_true",
        help="enable mixed precision computations",
    )
    tensorflow.add_argument(
        "--cudnn-determ",
        type=bool,
        default=True,
        help="set to False to disable CuDNN determinism",
    )
    tensorflow.add_argument(
        "--use-cuda-malloc-async",
        action="store_true",
        help="use CUDA malloc allocator for GPU",
    )
    tensorflow.add_argument(
        "--allow-memory-growth",
        action="store_true",
        help="allow GPU memory growth by not allocating all GPU memory preemptively",
    )

    rand = parser.add_argument_group("random")
    rand.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for NumPy and TensorFlow",
    )


def configure_tensorflow(
    log_level: int = 1,
    mixed_precision: bool = False,
    cudnn_determ: bool = True,
    use_cuda_malloc_async: bool = False,
    allow_memory_growth: bool = False,
) -> None:
    """Apply several TensorFlow configurations."""
    # This works even after importing tensorflow:
    # for i in 0 1; do python -c "import os, tensorflow as tf;      `# noqa:E800` \
    #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '$i';                `# noqa:E800` \
    #     tf.keras.models.Sequential([]); print('Done ($i)')"; done  # noqa:E800
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(log_level)

    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    if cudnn_determ:
        tf.config.experimental.enable_op_determinism()
        if os.name == "nt":
            # https://github.com/tensorflow/tensorflow/issues/39751
            os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"

    if use_cuda_malloc_async:
        # https://github.com/tensorflow/tensorflow/issues/48545
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    if allow_memory_growth:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def set_seeds(seed: int = 0) -> None:
    """Set Python, NumPy and TensorFlow seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    keras.utils.set_random_seed(seed)


def configure(args: Namespace) -> None:
    """Configure TensorFlow and random seeds."""
    configure_tensorflow(**args.groups.tensorflow)
    # Set seeds *after* configuring TensorFlow!
    set_seeds(args.seed)
