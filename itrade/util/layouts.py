"""Handle plate layouts."""
from __future__ import annotations

import re
import tempfile
from functools import total_ordering
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from . import itrex
from . import toolbox as tb

LAYOUT_COLS = ["Plate", "Row", "Column", "WellType", "Treatment", "Concentration"]
LAYOUT_XLSX_FILE_NAME = "iTReX-Demo_MRA_Layout-Imaging-3Plates.xlsx"
LAYOUT_CSV_FILE_NAME = "Layout_DS.csv"

PLATE_FORMAT = "P{}"

LAYOUT_TYPE_PATTERN = r"(?<=_)[A-Z]+(?=\d?$)"

COORD_COLS = ["Plate", "Row", "Column"]
COORD_STR_TEMPLATE = "{Plate}_{Row}{Column:02d}"

DUPL_COLS = ["Treatment", "Concentration"]
DUPL_SUFFIX = "Dupl"
COORD_COLS_DUPL = [c + DUPL_SUFFIX for c in COORD_COLS]

# Storing controls in this order implies the index matches the expected viability:
# "pos": index 0, viability 0%; "neg": index 1, viability 100%
# In Phase 3, percent viabilities fit nicely between these controls.
# This implies that after prediction, we use prediction[1] as the viability.
WELL_TYPES_CONTROLS = ("pos", "neg")
WELL_TYPE_THERAPY = "therapy"

VIABILITY_SCALE = 1000


def convert_itrex_ds_layout() -> None:
    """Convert from iTReX .xlsx layout to iTRADE .csv layout."""
    # > from itrade.util.layouts import convert_itrex_ds_layout as conv; conv()
    with tempfile.TemporaryDirectory() as tempdir:
        layout_xlsx = itrex.download_demo_file(LAYOUT_XLSX_FILE_NAME, Path(tempdir))
        df = pd.read_excel(layout_xlsx)

    df = df[LAYOUT_COLS]
    df.Plate = df.Plate.apply(PLATE_FORMAT.format)
    df.to_csv(LAYOUT_CSV_FILE_NAME, index=False)


def load_layout(layout_dir: Path, dataset: str) -> tuple[pd.DataFrame, str]:
    """Initialize from CSV layout file."""
    layout_match = re.search(LAYOUT_TYPE_PATTERN, dataset)
    if not layout_match:
        raise ValueError
    layout_type = layout_match[0]

    layout_basename = f"Layout_{layout_type}.csv"
    layout_file = layout_dir / layout_basename
    tb.ensure(layout_file.is_file(), "Layout file could not be found.")

    converters: dict[str, Callable[[str], Plate] | Callable[[str], int]] = {
        "Plate": Plate,
        "Column": int,
    }
    df: pd.DataFrame = pd.read_csv(layout_file, converters=converters, comment="#")
    df["CoordString"] = df.apply(lambda row: COORD_STR_TEMPLATE.format(**row), axis=1)
    df["IsTherapy"] = df.WellType == WELL_TYPE_THERAPY
    df["Label"] = df.WellType.apply(
        # 0 = pos, 1 = neg
        lambda x: WELL_TYPES_CONTROLS.index(x)
        if x in WELL_TYPES_CONTROLS
        else pd.NA
    )

    return df, layout_type


def merge_duplicate_wells(df: pd.DataFrame) -> pd.DataFrame:
    """Merge therapy wells onto their own duplicates."""
    df_therapy = df.loc[df.IsTherapy, DUPL_COLS + COORD_COLS + ["Viability"]]
    df = df.merge(df_therapy, how="outer", on=DUPL_COLS, suffixes=("", DUPL_SUFFIX))

    coords = df[COORD_COLS].rename(columns=lambda s: str(s) + DUPL_SUFFIX)
    coords_dupl = df[COORD_COLS_DUPL]
    df = df[coords.ne(coords_dupl).any(axis=1)]

    return df


@total_ordering
class Plate:
    """A plate represented by a 0-based int or a 1-based str, ensuring 0 == '1'."""

    @staticmethod
    def range(n_plates: int) -> map:
        """Return a range of plates."""
        return map(Plate, range(n_plates))

    # Order from https://docs.python.org/3/reference/datamodel.html#basic-customization

    def __init__(self, plate: int | str | Plate):
        """Initialize from either int (0-based) or str (1-based)."""
        if isinstance(plate, int):
            self.plate = plate
        elif isinstance(plate, str):
            # Extract "1" from "1" as well as "P1":
            matches = re.search(r"(?:(?<=^P)|^)\d+$", plate)
            if not matches:
                raise ValueError("unexpected Plate input string.")
            self.plate = int(matches[0]) - 1
        elif isinstance(plate, Plate):
            self.plate = plate.plate
        else:
            raise NotImplementedError("unsupported Plate input type.")
        if self.plate < 0:
            raise ValueError("negative Plate index.")

    def __repr__(self) -> str:
        """Return unambiguous representation."""
        return f"Plate('{self.plate + 1}')"

    def __str__(self) -> str:
        """Return human interpretation ('P1', 'P2', ...)."""
        return f"P{self.plate + 1}"

    def __eq__(self, other: Any) -> bool:
        """Determine equality based on representing the same plate."""
        if isinstance(other, Plate):
            return self.plate == other.plate

        try:
            return self == Plate(other)
        except ValueError:
            pass

        try:
            return self.plate == other
        except TypeError:
            return False

    def __gt__(self, other: Any) -> bool:
        """Provide comparison for @total_ordering."""
        if isinstance(other, Plate):
            return self.plate > other.plate

        try:
            return self > Plate(other)
        except ValueError:
            pass

        try:
            return self.plate > other
        except TypeError:
            return False

    def __hash__(self) -> int:
        """Return hash."""
        return hash(self.plate)

    def __add__(self, other: int) -> Plate:
        """Return a new plate shifted by +other."""
        return Plate(self.plate + other)

    def __sub__(self, other: int | Plate) -> Plate | int:
        """Return the distance between two plates, or a new plate shifted by -other."""
        if isinstance(other, Plate):
            return self.plate - other.plate
        return Plate(self.plate - other)

    def __int__(self) -> int:
        """Convert by int by returning plate."""
        return self.plate

    def __index__(self) -> int:
        """Use object to index lists."""
        return self.plate
