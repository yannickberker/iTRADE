"""Run a CNN."""
import argparse
import sys
from pathlib import Path
from typing import Any

from .cnn import bye, configuration, datasets, export, models, operations, visualization
from .util import arggroupparse


def mkdir(in_path: str) -> Path:
    """Path type for argparse."""
    path = Path(in_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    ops = parser.add_argument_group("operations")
    ops.add_argument(
        "--load-model",
        action="store_true",
        help="load model",
    )
    ops.add_argument(
        "--load-model-file-name",
        type=Path,
        default="model.tf",
        help="load model from this file",
    )
    ops.add_argument(
        "--vis-model",
        action="store_true",
        help="visualize model",
    )
    ops.add_argument(
        "--no-train",
        action="store_false",
        help="load model from --save(!)-model-file-name and skip training",
        dest="train_model",
    )
    ops.add_argument(
        "--val-model",
        "--validate-model",
        "--validate",
        action="store_true",
        help="validate model during training using validation data",
    )
    ops.add_argument(
        "--save-model",
        "--save",
        action="store_true",
        help="save model to file after training",
    )
    ops.add_argument(
        "--predict-model",
        "--predict",
        action="store_true",
        help="make a prediction using test data",
    )
    ops.add_argument(
        "--test-model",
        "--test",
        action="store_true",
        help="evaluate prediction using labeled test or validation data",
    )
    ops.add_argument(
        "--export-data",
        action="store_true",
        help="export data pre training and post prediction to pptx file",
    )
    ops.add_argument(
        "--export-gen",
        action="store_true",
        help="export generators during training and validation to png file",
    )
    ops.add_argument(
        "--export-roc",
        "--test-export-roc",
        action="store_true",
        help="export ROCs post prediction using training/validation/testing data",
    )
    ops.add_argument(
        "--export-tsne",
        "--test-export-tsne",
        action="store_true",
        help="export t-SNE embeddings post prediction using labeled testing data",
    )
    ops.add_argument(
        "--results-dir",
        type=mkdir,
        default="results",
        help="output data to this directory",
    )
    ops.add_argument(
        "--save-model-file-name",
        type=Path,
        default="model.tf",
        help="output saved model to this file",
    )
    ops.add_argument(
        "--run-id",
        type=str,
        help="run ID; omit to be prompted for one",
    )


def process_args(*args_any: Any) -> argparse.Namespace:
    """Use argparse to parse arguments (as in https://redd.it/3do2wr, ct7n98v)."""
    parser = arggroupparse.ArgumentParser(description="Run a CNN")

    add_arguments(parser)
    configuration.add_arguments(parser)
    datasets.add_arguments(parser)
    models.add_arguments(parser)
    operations.add_arguments(parser)

    if args_any:
        # Convert all original argument values to strings
        args_str = [run_cnn.__name__, *map(str, args_any)]
    else:
        # Use command-line arguments
        args_str = sys.argv

    # Remove falsy argument values such as "" (https://stackoverflow.com/a/53744025)
    args_str = list(filter(None, args_str[1:]))

    if not args_str:
        parser.print_help(sys.stderr)

    args = parser.parse_args(args_str)

    args.args_str = args_str

    if not args.run_id:
        args.run_id = input("Input run ID: ") or "EmptyRunID-" + args.dataset

    if args.fast_try:
        args.run_id = "FastTry-" + args.run_id

    if args.load_model and not args.train_model:
        # When rerunning CNN predictions, load the previous final (target) model ...
        args.load_model_file_name = args.save_model_file_name
        # ... and prevent unnecessary overwriting.
        args.save_model = False

    args.run_dir = args.results_dir / args.run_id

    return args


def run_cnn(*args_any: Any) -> Path:
    """Train and run network."""
    args = process_args(*args_any)

    print(f"Starting run '{args.run_id}'.")

    configuration.configure(args)

    # Load data
    data = datasets.load(args)

    # Create model
    if args.load_model:
        model = models.load_model(args, data)
    else:
        model = models.create_model(args, data)

    if args.train_model:
        if args.export_data:
            export.before_training(args, data)

        model = operations.train_model(args, data, model)

    # Use model
    if args.save_model:
        models.save_model(args, model)

    if args.predict_model:
        operations.predict_model(args, data, model)

        if args.vis_model:
            visualization.vis_model(args, data, model)

        if args.export_data:
            export.after_prediction(args, data)

        if args.export_tsne:
            export.tsne(args, data)

        if args.export_roc:
            export.roc(args, data)

    bye.bye()

    return args.run_dir


if __name__ == "__main__":
    run_cnn()
