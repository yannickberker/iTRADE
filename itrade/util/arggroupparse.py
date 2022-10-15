"""Improve argparse using private APIs."""
import argparse
import warnings
from argparse import Namespace
from typing import Any, Sequence


class WideMetavarDefaultsFormatter(
    argparse.MetavarTypeHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Help formatter safely using max_help_position private API."""

    def __init__(self, prog: str, max_help_position: int = 50):
        """Initialize using width and max_help_position, if possible."""
        try:
            # "Only the name of this class is considered a public API"
            # (https://github.com/python/cpython/blob/v3.10.2/Lib/argparse.py#L154-L159)
            super().__init__(prog, max_help_position=max_help_position)
        except TypeError:
            warnings.warn("WideMetavarDefaultsFormatter failed, falling back.")
            super().__init__(prog)


class ArgumentParser(argparse.ArgumentParser):
    """Argument parser with wider help, also returning args.groups."""

    def __init__(self, **kwargs: Any):
        """Call super with wider HelpFormatter."""
        super().__init__(**kwargs, formatter_class=WideMetavarDefaultsFormatter)

    # We could copy all @overload's for ArgumentParser.parse_args here,
    # but we do not allow arbitrary _N anyway since we assign .groups:
    def parse_args(  # type: ignore[override]
        self, args: Sequence[str] = None, namespace: Namespace = None
    ) -> Namespace:
        """Call parse_args and extract arguments by groups."""
        parsed_args = super().parse_args(args, namespace)

        parsed_args.groups = Namespace(
            **{
                # pylint: disable=protected-access
                group.title: {
                    action.dest: getattr(parsed_args, action.dest, None)
                    for action in group._group_actions
                }
                for group in self._action_groups
                if group.title is not None
            }
        )

        return parsed_args
