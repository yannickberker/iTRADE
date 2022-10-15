"""Loading data sets for CNN."""
import sys
from argparse import ArgumentParser, Namespace
from itertools import compress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from tensorflow import keras

from ..util import images, layouts
from ..util import toolbox as tb
from ..util.images import KERAS_MAX
from . import keras_app

# Don't confuse with Keras operations:
# - Keras "fit" uses TRAIN and VAL data
# - Keras "predict" uses PREDICT data
# Our testing uses PREDICT data (if it has labels) or falls back to VAL data
TRAIN = "train"
VAL = "validation"
PREDICT = "predict"
DATA_TYPES = OrderedSet((TRAIN, VAL, PREDICT))

PROJ = "proj"
MIDZ = "midz"
ALL_IMAGE_TYPES = OrderedSet((PROJ, MIDZ))

if sys.gettrace():
    pd.options.display.width = 0


def add_arguments(parser: ArgumentParser) -> None:
    """Add data arguments."""
    metadata = parser.add_argument_group("metadata")
    metadata.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="data directory",
    )
    metadata.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="data set to use (screen ID)",
    )
    metadata.add_argument(
        "--channels",
        nargs=3,
        default=[PROJ, MIDZ, PROJ],
        type=str,
        choices=ALL_IMAGE_TYPES,
        help="RGB image channel composition (three-element list)",
    )
    metadata.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=keras_app.IMAGENET_IMG_SHAPE,
        help="image target shape (two ints)",
    )
    metadata.add_argument(
        "--val-samples",
        type=int,
        default=0,
        help="number of validation samples per plate",
    )
    metadata.add_argument(
        "--fast-try",
        action="store_true",
        help="enable fast try for debugging (6 samples, 2 epochs)",
    )
    metadata.add_argument(
        "--ct-files",
        type=Path,
        help=(
            "names of text files for Phase-3 cross-training, "
            "using {P} as a placeholder for plates ('P1', 'P2', ...)"
        ),
    )

    processing = parser.add_argument_group("processing")
    processing.add_argument(
        "--shift-images",
        action="store_true",
        help="shift the mean of float images up by 0.5 before Keras preprocessing",
    )
    processing.add_argument(
        "--center-images",
        action="store_true",
        help="set the mean of float images to 0.5 before Keras preprocessing",
    )
    processing.add_argument(
        "--mix-images",
        action="store_true",
        help="use image-mixing data augmentation for moldev_imagenet_class",
    )
    processing.add_argument(
        "--skip-da",
        "--skip-data-augmentation",
        action="store_true",
        help="skip data augmentation during training",
    )


def load_metadata(
    data_dir: Path,
    dataset: str,
    channels: list[str],
    shape: tuple[int, ...],
    val_samples: int,
    fast_try: bool,
    ct_files: Path,
) -> dict[str, Any]:
    """Load filenames, class names, and labels."""
    layout_dir = data_dir / "Layouts"
    tb.ensure(layout_dir.is_dir(), "Layouts folder could not be found")

    (wells, layout_type) = layouts.load_layout(layout_dir, dataset)

    image_dir = data_dir / "Images" / dataset
    tb.ensure(image_dir.is_dir(), "Image folder could not be found")

    image_types = ALL_IMAGE_TYPES.intersection(channels)

    # Generate file names
    prefixes = dataset + "_" + wells.CoordString + "_"
    suffix = "_" + "x".join(map(str, shape)) + ".tif"
    for image_type in image_types:
        basenames = prefixes + image_type + suffix
        filenames = image_dir / basenames
        filenames = [f if f.is_file() else pd.NA for f in filenames]
        wells["File_" + image_type] = filenames

    wells["HasFiles"] = ~wells[("File_" + it for it in image_types)].isna().any(1)

    # By default (Phase 1/2), include only control wells in training/validation
    well_types_tv = set(layouts.WELL_TYPES_CONTROLS)
    if ct_files:
        # By contrast, in Phase 3, also include therapy wells in training
        well_types_tv.add(layouts.WELL_TYPE_THERAPY)

    wells["IsTV"] = wells.WellType.isin(well_types_tv)
    wells_tv = wells[wells.IsTV & wells.HasFiles]

    if fast_try:
        # Make sure we keep all kinds of controls (positive, negative, dupl. therapy):
        # keep the bottom two non-empty border rows on the first plate ...
        one_plate = wells.Plate.iloc[0]
        two_rows = wells.Row[wells.HasFiles].unique()[[-1, -2]]
        wells = wells[(wells.Plate == one_plate) & wells.Row.isin(two_rows)]
        wells_tv = wells_tv[(wells_tv.Plate == one_plate) & wells_tv.Row.isin(two_rows)]
        # ... and reduce the number of validation controls to the minimum.
        val_samples = 2

    plates = set(wells_tv.Plate)

    if ct_files:
        # Phase 3
        if layout_type == "CE":
            raise NotImplementedError("Cross-training implemented only for drug wells.")

        dfs = []
        for plate in plates:
            ct_file = next(ct_files.parent.glob(ct_files.name.format(P=plate)))
            df = pd.read_csv(ct_file, delimiter="\t", index_col=0)
            df.columns = df.columns.astype(int)
            df = df.melt(var_name="Column", value_name="Viability", ignore_index=False)
            df = df.rename_axis("Row").reset_index()
            df = df.assign(Plate=plate)
            dfs.append(df)

        # 0.1 = more pos, 0.9 = more neg
        ct_viab = pd.concat(dfs)
        ct_viab.Viability /= layouts.VIABILITY_SCALE
        wells_tv = wells_tv.merge(ct_viab, on=layouts.COORD_COLS)

        wells_tv = layouts.merge_duplicate_wells(wells_tv)

        viab_dupl = wells_tv[wells_tv.IsTherapy]["Viability" + layouts.DUPL_SUFFIX]
        wells_tv.loc[wells_tv.IsTherapy, "Label"] = viab_dupl

    # Check training/validation labels
    tb.ensure(not any(wells_tv["Label"].isna()), "Labels incomplete")

    if layout_type == "CE":  # for backwards compatibility
        sort_keys = ["IsTherapy", "Plate", "WellType", "Row", "Column"]
    else:
        sort_keys = ["IsTherapy", "Plate", "WellType", "Column", "Row"]
    wells_tv = wells_tv.sort_values(sort_keys)

    n_plates = len(plates)
    n_classes = len(layouts.WELL_TYPES_CONTROLS)
    tb.ensure(val_samples % (n_plates * n_classes) == 0)

    # per plate and class
    max_val = val_samples // (n_classes * n_plates)
    n_val = {p: np.zeros(n_classes) for p in plates}

    # Initialize metadata dictionary
    data: dict[str, Any] = {"layout": wells, "classes": layouts.WELL_TYPES_CONTROLS} | {
        dt: {**{it: [] for it in image_types}, "coord": []} for dt in DATA_TYPES
    }
    data[TRAIN]["label"] = []
    data[VAL]["label"] = []

    # Fill metadata dictionary
    for image_type in image_types:
        data[PREDICT][image_type] = wells[wells.HasFiles]["File_" + image_type].tolist()
    data[PREDICT]["coord"] = wells[wells.HasFiles].CoordString.tolist()

    for _, well in wells_tv.iterrows():
        # store first max_val T/V wells from each plate and class for validation data
        add_to_val = (not well.IsTherapy) and (n_val[well.Plate][well.Label] < max_val)
        data_type = VAL if add_to_val else TRAIN
        if add_to_val:
            # Phase 1 (control experiments, _CE):
            # - Class 0 (STS):    B16-B23, C16-C17
            # - Class 1 (DMSO):   B05-B11, C05-C07

            # Phase 2/3 (drug screens, _DS):
            # - Class 0 (STS):    M04 (P1-P3)
            # - Class 1 (DMSO):   D05 (P1-P3)
            n_val[well.Plate][well.Label] += 1

        for image_type in image_types:
            data[data_type][image_type].append(well["File_" + image_type])
        data[data_type]["label"].append(well.Label)
        data[data_type]["coord"].append(well.CoordString)

    return data


def load(args: Namespace) -> dict[str, Any]:
    """Load (train|val|predict)_(input|label)(_ori) data and a few more."""
    data = load_metadata(**args.groups.metadata)

    dt_args = [a or args.predict_model for a in (args.train_model, args.val_model, 0)]
    data_types = list(compress(DATA_TYPES, dt_args))
    image_types: OrderedSet = OrderedSet(args.channels)

    # convert classes to categorical
    class_names = data["classes"]
    n_classes = len(class_names)

    def to_categorical(group_classes: np.ndarray) -> np.ndarray:
        group_labels = keras.utils.to_categorical(group_classes, num_classes=n_classes)
        mixed_classes = [group_class not in {0.0, 1.0} for group_class in group_classes]
        if any(mixed_classes):
            # Mixed classes are supported only for two classes
            tb.ensure(n_classes == 2)

            # Smooth (only) hard labels manually:
            factor = args.label_smoothing
            # Disable downstream label smoothing (in the loss function)
            args.label_smoothing = 0
            # https://github.com/keras-team/keras/blob/v2.8.0/keras/losses.py#L1784
            group_labels = group_labels * (1.0 - factor) + (factor / n_classes)

            # Mixed classes are assumed smoothed (Phase-2 outputs)
            mixed_class_labels = np.array(group_classes)[mixed_classes]
            # As they are prediction[1], convert them back to two elements:
            group_labels[mixed_classes, 0] = 1 - mixed_class_labels
            group_labels[mixed_classes, 1] = 0 + mixed_class_labels

        return group_labels

    # mix images
    def mix_augmentation(
        inputs: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        class0 = np.where(labels[:, 0])[0]
        class1 = np.where(labels[:, 1])[0]
        n_mix = min(len(class0), len(class1))

        im_shape = inputs.shape[1:]
        mix_images = np.nan * np.empty([2 * n_mix, *im_shape], dtype=inputs.dtype)
        mix_labels = np.nan * np.empty([2 * n_mix, 2], dtype=labels.dtype)
        alphas = np.random.random_sample(size=n_mix)  # 1 means use class 0
        dims = np.random.randint(2, size=n_mix)
        for i, (c_0, c_1, alpha, dim) in enumerate(zip(class0, class1, alphas, dims)):
            lim = np.uint16(alpha * im_shape[dim])
            idx_low = [slice(None)] * 3
            idx_high = [slice(None)] * 3
            idx_low[dim] = slice(lim)
            idx_high[dim] = slice(lim, None)
            mix_images[(2 * i, *idx_low)] = inputs[(c_0, *idx_low)]
            mix_images[(2 * i, *idx_high)] = inputs[(c_1, *idx_high)]
            mix_labels[2 * i] = [alpha, 1 - alpha]

            mix_images[2 * i + 1] = alpha * inputs[c_0] + (1 - alpha) * inputs[c_1]
            mix_labels[2 * i + 1] = [alpha, 1 - alpha]

        tb.ensure(not np.any(np.isnan(mix_images)))
        all_images = np.concatenate((inputs, mix_images))
        labels = np.concatenate((labels, mix_labels))
        return (all_images, labels)

    # convert from image types to channels
    def types_to_channels(data: np.ndarray) -> np.ndarray:
        return np.stack(
            [data[:, :, :, image_types.index(c)] for c in args.channels], axis=-1
        )

    for dt in data_types:
        if labels := data[dt].get("label"):
            # check label range
            tb.ensure(all(0.0 <= c <= 1.0 for c in labels), "label range")

            # convert to categorical
            data[dt]["label"] = to_categorical(labels)

        # read images
        files = [data[dt][it] for it in image_types]
        if files == [[]] * len(image_types):
            continue
        data[dt]["images"] = images.file2gray(zip(*files), target_size=args.shape)

        # check image range
        tb.ensure(0.0 <= np.min(data[dt]["images"]) <= 0.5 * KERAS_MAX, "image min")
        tb.ensure(0.0 <= np.max(data[dt]["images"]) <= 1.0 * KERAS_MAX, "image max")

        if dt == TRAIN and args.mix_images:
            (data[dt]["images"], data[dt]["label"]) = mix_augmentation(
                data[dt]["images"], data[dt]["label"]
            )

        data[dt]["input_ori"] = types_to_channels(data[dt]["images"])
        data[dt]["input"] = data[dt]["input_ori"].copy()

        # Empirical shifts to account for heavy-right-tailed intensities (mean ~= 0)
        if args.shift_images:
            data[dt]["input"] += KERAS_MAX / 2

        if args.center_images:
            data[dt]["input"] += KERAS_MAX / 2 - np.mean(data[dt]["input"])

        # We only ever handle RGB data. Exceptions:
        # 1. Keras preprocess_input will convert to BGR for *some* models (e.g., VGG16):
        #    https://github.com/keras-team/keras/blob/v2.8.0/keras/applications/...
        #    .../vgg16.py#L230-L233, .../imagenet_utils.py#L69-L80, and
        #    https://github.com/keras-team/keras-applications/issues/21
        data[dt]["input"] = keras_app.preprocess_input(args.model, data[dt]["input"])

        # 2. cv2.imencode expects BGR images in itrade.util.images.rgb2png_buffer
        #    https://stackoverflow.com/a/52495126/880783
        #    We pass data[dt]["input_ori"] which is RGB, so we need to handle it there.

        # 3. tf.summary.image in InputImageCallback expects RGB images.
        #    https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/...
        #    .../python/summary/summary.py#L119
        #    It sees data[df]["input"] which can be either format depending on the
        #    model (see 1). We current ignore this when exporting to TensorBoard.

        if dt == TRAIN and not args.skip_da:
            da_kwargs = {
                "rotation_range": 30,
                "width_shift_range": 0.1,
                "height_shift_range": 0.1,
                "brightness_range": [0.5, 1.5],
                "fill_mode": "reflect",
                "zoom_range": 0.1,
                "horizontal_flip": True,
                "vertical_flip": True,
            }
            flow_kwargs = {
                "shuffle": True,
                "seed": args.seed,
            }
        else:
            da_kwargs = {}
            flow_kwargs = {}

        data[dt]["datagen"] = keras.preprocessing.image.ImageDataGenerator(**da_kwargs)

        data[dt]["flow"] = data[dt]["datagen"].flow(
            data[dt]["input"],
            data[dt].get("label"),
            batch_size=args.batch_size,
            **flow_kwargs,
        )

    # Set data shapes
    for data_type in DATA_TYPES:
        try:
            data["in_shape"] = data[data_type]["input"].shape[1:]
            break
        except KeyError:
            pass
    data["out_shape"] = (len(data["classes"]),)

    if not args.train_model:
        return data

    data["fit"] = {
        "x": data[TRAIN]["flow"],
        "steps_per_epoch": len(data[TRAIN]["flow"]),
    }

    if args.val_model:
        data["fit"] |= {
            "validation_data": data[VAL]["flow"],
            "validation_steps": len(data[VAL]["flow"]),
        }

    return data
