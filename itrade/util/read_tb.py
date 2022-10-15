"""Read metrics from TensorBoard event file."""
import json
import sys
from typing import Any, Literal, Sequence

import tensorflow as tf


def proto_to_float(tensor: Any) -> float:
    """Convert a 0-D Tensor Proto to a float."""
    return float(tf.make_ndarray(tensor))


def read_from_file_tb(exp_steps: int, event_file: str) -> dict[str, list[float]]:
    """See `read_from_file`."""
    # pylint: disable=import-outside-toplevel
    from tensorboard.backend.event_processing import event_accumulator, tag_types

    results: dict[str, list[float]] = {}

    accumulator = event_accumulator.EventAccumulator(
        event_file, size_guidance={tag_types.TENSORS: exp_steps}
    )
    accumulator.Reload()
    for tag in accumulator.tensors.Keys():
        if tag.startswith("epoch_"):
            events = accumulator.Tensors(tag)
            if exp_steps > 0:
                if len(events) != exp_steps:
                    raise ValueError
                events = [events[-1]]
            results[tag] = [proto_to_float(e.tensor_proto) for e in events]

    return results


def read_from_file_tf(exp_steps: int, event_file: str) -> dict[str, list[float]]:
    """See `read_from_file`."""
    # pylint: disable=import-outside-toplevel,import-error
    from tensorflow.compat.v1.train import (  # pyright: ignore[reportMissingImports]
        summary_iterator,
    )

    results: dict[str, list[float]] = {}

    counts: dict[str, int] = {}
    for summary in summary_iterator(event_file):
        for event in summary.summary.value:
            if (tag := event.tag).startswith("epoch_"):
                value = proto_to_float(event.tensor)
                counts[tag] = counts.get(tag, 0) + 1
                if counts[tag] == 1:
                    results[tag] = []
                if exp_steps in (0, counts[tag]):
                    results[tag].append(value)
    if exp_steps > 0 and any(count != exp_steps for count in counts.values()):
        raise ValueError

    return results


def read_from_file(
    exp_steps: int,
    event_file: str,
    backend: Literal["tensorboard", "tensorflow"] = "tensorboard",
) -> dict[str, list[float]]:
    """Process metrics from a single event file using one of two private backends.

    If `exp_steps` equals 0, return metrics for all steps (usually epochs). Otherwise,
    ensure we have exactly `exp_steps` steps and return metrics only for the final step.
    """
    if backend == "tensorboard":
        return read_from_file_tb(exp_steps, event_file)
    if backend == "tensorflow":
        return read_from_file_tf(exp_steps, event_file)
    raise ValueError


def read_from_files(exp_steps: int, event_files: Sequence[str]) -> None:
    """Collect metrics from many event files."""
    results = [read_from_file(exp_steps, event_file) for event_file in event_files]
    print(json.dumps(results))


if __name__ == "__main__":
    read_from_files(int(sys.argv[1]), sys.argv[2:])
