"""Wrapper for keras.application models."""
from typing import Callable, TypeAlias

import numpy as np
from tensorflow import keras

from ..util import toolbox as tb

# from tensorflow.keras import layers  # noqa: E800
# https://github.com/tensorflow/tensorflow/pull/54104#issuecomment-1067102133
# https://github.com/tensorflow/tensorflow/pull/54104#issuecomment-1077827486
layers = keras.layers

Layer: TypeAlias = keras.layers.Layer

IMAGENET_IMG_SHAPE = (224, 224)
IMAGENET_N_CLASSES = 1000

MODELS = {
    # "class_name": "module_name"  # input_shape, exclusion criterion
    "DenseNet121": "densenet",  # 224
    "DenseNet169": "densenet",  # 224
    "DenseNet201": "densenet",  # 224
    # "EfficientNetB0": "efficientnet",  # 224, no add_top_layers
    # "EfficientNetB1": "efficientnet",  # 240, != 224
    # "EfficientNetB2": "efficientnet",  # 260, != 224
    # "EfficientNetB3": "efficientnet",  # 300, != 224
    # "EfficientNetB4": "efficientnet",  # 380, != 224
    # "EfficientNetB5": "efficientnet",  # 456, != 224
    # "EfficientNetB6": "efficientnet",  # 528, != 224
    # "EfficientNetB7": "efficientnet",  # 600, != 224
    # "EfficientNetV2B0": "efficientnet_v2",  # 224, no add_top_layers
    # "EfficientNetV2B1": "efficientnet_v2",  # 240, != 224
    # "EfficientNetV2B2": "efficientnet_v2",  # 260, != 224
    # "EfficientNetV2B3": "efficientnet_v2",  # 300, != 224
    # "EfficientNetV2S": "efficientnet_v2",  # 384, != 224
    # "EfficientNetV2M": "efficientnet_v2",  # 480, != 224
    # "EfficientNetV2L": "efficientnet_v2",  # 480, != 224
    # "InceptionResNetV2": "inception_resnet_v2",  # 299 != 224
    # "InceptionV3": "inception_v3",  # 299 != 224
    # "MobileNet": "mobilenet",  # 224, no add_top_layers
    "MobileNetV2": "mobilenet_v2",  # 224
    # "MobileNetV3Small": "mobilenet_v3",  # 224, no add_top_layers
    # "MobileNetV3Large": "mobilenet_v3",  # 224, no add_top_layers
    # "NASNetMobile": "nasnet",  # 331, != 224
    # "NASNetLarge": "nasnet",  # 331, != 224
    "ResNet50": "resnet",  # 224
    "ResNet101": "resnet",  # 224
    "ResNet152": "resnet",  # 224
    "ResNet50V2": "resnet_v2",  # 224
    "ResNet101V2": "resnet_v2",  # 224
    "ResNet152V2": "resnet_v2",  # 224
    "VGG16": "vgg16",  # 224
    "VGG19": "vgg19",  # 224
    # "Xception": "xception",  # 299, != 224
}

DENSE_LAYER_MODEL_PREFIXES = ("VGG",)
DENSE_LAYERS = {
    "VGG16": (4096, 4096),
    "VGG19": (4096, 4096),
}
AVG_POOL_MODEL_PREFIXES = ("DenseNet", "MobileNetV2", "NASNet", "ResNet", "Xception")


def get_class(class_name: str) -> Callable:
    """Get model class for model class name."""
    class_ = getattr(keras.applications, class_name)
    return class_


def preprocess_input(class_name: str, images: np.ndarray) -> np.ndarray:
    """Preprocess images according to model class name."""
    module = getattr(keras.applications, MODELS[class_name])
    return module.preprocess_input(images)


def add_top_layers(layer: Layer, class_name: str, n_classes: int) -> Layer:
    """Return include_top=False model with top layers added back on top."""
    # https://github.com/keras-team/keras/blob/v2.8.0/keras/applications/...
    if class_name.startswith(DENSE_LAYER_MODEL_PREFIXES):
        # .../vgg16.py#L185-L193
        # .../vgg19.py#L190-L197
        layer = layers.Flatten(name="flatten")(layer)
        for i, d_l in enumerate(DENSE_LAYERS[class_name], 1):
            layer = layers.Dense(d_l, activation="relu", name=f"fc{i}")(layer)
    elif class_name.startswith(AVG_POOL_MODEL_PREFIXES):
        # .../densenet.py#L248-L253
        # .../mobilenet_v2.py#L388-L392
        # .../nasnet.py#L262-L266
        # .../resnet.py#L172-L176
        # .../xception.py#L276-L280
        layer = layers.GlobalAveragePooling2D(name="avg_pool")(layer)
    else:
        # .../efficientnet.py#L372-L381
        # .../efficientnet_v2.py#L1018-L1028
        # .../mobilenet.py#L262-L269
        # .../mobilenet_v3.py#L313-L321
        tb.ensure(False, f"add_top_layers not implemented for {class_name}.")

    # https://www.tensorflow.org/guide/mixed_precision#building_the_model
    layer = layers.Dense(n_classes, name="dense_logits")(layer)
    layer = layers.Activation("softmax", dtype="float32", name="predictions")(layer)

    return layer


def keras_app_model(
    model: str,
    weights: str = "imagenet",
    input_shape: tuple[int, ...] = (*IMAGENET_IMG_SHAPE, 3),
    n_classes: int = IMAGENET_N_CLASSES,
) -> keras.models.Model:
    """Return a keras.applications model of an (ImageNet-pretrained) network."""
    class_ = get_class(model)

    # we can safely include_top if we don't need ImageNet weights;
    # otherwise, we need input_shape and n_classes to match those of the weights.
    include_top = weights != "imagenet" or (
        input_shape == (*IMAGENET_IMG_SHAPE, 3) and n_classes == IMAGENET_N_CLASSES
    )

    base_model = class_(
        input_shape=input_shape,
        include_top=include_top,
        weights=weights,
        classes=n_classes,
    )

    if include_top:
        return base_model

    layer = base_model.output
    layer = add_top_layers(layer, model, n_classes)
    model = keras.models.Model(inputs=base_model.input, outputs=layer)
    return model
