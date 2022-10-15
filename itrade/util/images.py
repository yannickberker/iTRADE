"""Handle image files."""
import io
import sys
from collections.abc import Iterable, Sequence
from os import PathLike
from pathlib import Path
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
from colour import XYZ_to_sRGB
from colour.colorimetry import wavelength_to_XYZ

from . import pptx
from . import toolbox as tb

# https://github.com/keras-team/keras/blob/v2.8.0/keras/applications/imagenet_utils.py
KERAS_MAX = np.iinfo(np.uint8).max

HOECHST_WAVELENGTH = 461


def file2gray(
    filename: str | Path | Iterable[str | Path],
    target_size: list[int] = None,
    axis: int = 0,
) -> np.ndarray:
    """Read (and resize) an image file into a [0, KERAS_MAX] Keras image."""
    tb.ensure(filename is not None)

    if not isinstance(filename, (str, Path)):
        tb.ensure(isinstance(filename, Iterable))
        imgs = [file2gray(f, target_size=target_size, axis=-1) for f in filename]
        stack = np.stack(imgs, axis=axis)
        return stack

    # read raw values
    image_gray = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    raw_max = np.iinfo(image_gray.dtype).max

    # rescale to Keras expected range (before preprocessing)
    image_gray = image_gray.astype(np.float32) * (KERAS_MAX / raw_max)

    return image_gray


def gray2rgb(
    image: np.ndarray,
    scale: float = 1.0,
    wavelength: int = HOECHST_WAVELENGTH,
    colormode: Literal["wavelength", "rgb", "manual"] = "wavelength",
) -> np.ndarray:
    """Convert a grayscale NumPy array into a color NumPy array."""
    if colormode == "wavelength":
        if wavelength is None:
            color = np.array([1.0, 1.0, 1.0])  # white
        else:
            # https://aty.sdsu.edu/explain/optics/rendering.html
            # https://aty.sdsu.edu/explain/optics/color/color.html
            xyz = wavelength_to_XYZ(wavelength)
            color = XYZ_to_sRGB(xyz)
            color = color.clip(0, 1)
    elif colormode == "rgb":
        if wavelength is None:
            raise NotImplementedError
        color = np.array(
            [
                580 < wavelength <= 780,  # red
                490 < wavelength <= 580,  # green
                380 <= wavelength <= 490,  # blue
            ]
        )
        color = np.array(color).astype(float)
    elif colormode == "manual":
        # https://academo.org/demos/wavelength-to-colour-relationship/
        # https://gist.github.com/cad598361683b10c5bc6787aa9951d64
        if wavelength == HOECHST_WAVELENGTH:
            color = np.array([0.0, 0.5, 1.0])  # #007FFF
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    shape = (1,) * image.ndim + (3,)
    color = np.reshape(color, shape)

    image = np.expand_dims(image, -1)

    image_scaled = image * scale

    image_color = image_scaled * color

    image_color = rgb_scale_clip(image_color)

    return image_color


def rgb_scale_clip(image_rgb: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Scale to [0, KERAS_MAX], clipping values while ensuring colors remain correct."""
    image_out = image_rgb * scale

    # Divide each pixel by 1 or (if larger) the maximum channel value.
    image_out /= np.maximum(1, image_out.max(axis=-1, keepdims=True) / KERAS_MAX)

    # Smooth out numerical inaccuracies
    image_out = image_out.clip(0, KERAS_MAX)

    return image_out


def rgb2png_buffer(image_rgb: np.ndarray, scale: float = 10.0) -> io.BytesIO:
    """Convert a Keras RGB image [0, KERAS_MAX] to a PNG file stored in a buffer."""
    image_out = rgb_scale_clip(image_rgb, scale)

    _, buffer = cv2.imencode(".png", cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR))
    image_buffer = io.BytesIO(buffer)

    return image_buffer


def images2pptx(
    images: np.ndarray,
    coords: Iterable[str],
    filename: PathLike,
    labels: Sequence[float] = (),
    preds: Sequence[float] = (),
) -> None:
    """Export images and labels to PowerPoint."""
    if len(labels) == 0:
        labels = [np.nan] * len(images)
    if len(preds) == 0:
        preds = [np.nan] * len(images)
    contents = [
        pptx.TextAndImage(
            text=f"well: {coord}, label: {label:.2g}, prediction: {pred:.2g}",
            image=rgb2png_buffer(image),
        )
        for (image, coord, label, pred) in zip(
            images, coords, labels, preds, strict=True
        )
    ]

    try:
        pptx.write_powerpoint(contents, filename)
    except Exception:  # pylint:disable=broad-except
        print("Writing PowerPoint failed, but is non-essential. Continuing...")


def tiff2png(tiff: Path, png: Path, scale: float) -> None:
    """Convert TIFF raw-data image to PNG RGB paper visualization image."""
    gray = file2gray(tiff)
    rgb = gray2rgb(gray, scale=scale) / KERAS_MAX
    png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(png, rgb)


if __name__ == "__main__":
    tiff2png(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]))
