"""Toolbox."""
import datetime
import inspect
import re
import types
import warnings

import humanize

# Print elapsed times


def now() -> datetime.datetime:
    """Return reference time for functions such as print_elapsed."""
    return datetime.datetime.now()


def print_elapsed(
    start: datetime.datetime,
    task_name: str = "Task",
    nel: int = None,
    el_name: str = "element",
) -> None:
    """Pretty-print elapsed time for a task."""
    delta = now() - start

    def pretty(dlt: datetime.timedelta) -> str:
        return humanize.naturaldelta(dlt, minimum_unit="microseconds")

    print(
        f"{task_name} completed in {pretty(delta)}"
        + (f", that is {pretty(delta / nel)} per {el_name}" if nel else "")
        + "."
    )


# Pretty assert


def ensure(condition: bool, msg: str = "", warn_only: bool = False) -> None:
    """
    Raise error that will not be optimized away (like assert).

    Raise warning pointing to previous stack level. Exit if warn_only is False.
    """
    if condition:
        return
    if msg:
        thumbs(condition, msg)
    warnings.warn(msg, stacklevel=2)
    if warn_only:
        return

    # https://stackoverflow.com/a/58821552
    frame = inspect.currentframe()
    if frame is None:
        raise AssertionError()

    back_frame = frame.f_back
    if back_frame is None:
        raise AssertionError()

    back_tb = types.TracebackType(
        tb_next=None,
        tb_frame=back_frame,
        tb_lasti=back_frame.f_lasti,
        tb_lineno=back_frame.f_lineno,
    )
    raise AssertionError().with_traceback(back_tb)


def thumbs(flag: bool, text: str, action: str = "double-check") -> None:
    """Print 👍 or ❌ and some text."""
    if flag:
        thumb = "👍"
        text = re.sub(r"\[.*\]\s*", "", text)
    else:
        thumb = "❌"
        text = re.sub(r"[\[\]]", "", text) + (f", {action}!" if action else "")
    print(f"{thumb}: {text}")
