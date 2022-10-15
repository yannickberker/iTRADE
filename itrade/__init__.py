"""Initialize path and filename constants."""
import os
from pathlib import Path

PACKAGE_SHORT = "iT"
PACKAGE_NAME = "iTRADE"

DIR_ENV_VAR = PACKAGE_NAME + "_DIR"
BASE_DIR = Path(os.getenv(DIR_ENV_VAR, default=os.getcwd())).expanduser().resolve()

DATA_DIR = BASE_DIR / (PACKAGE_NAME.lower() + "-data")
RESULTS_DIR = BASE_DIR / (PACKAGE_NAME.lower() + "-results")
ITREX_DIR = BASE_DIR / "itrex-results"
PLOTS_DIR = BASE_DIR / "plot-results"

DATA_ZIP_NAME = PACKAGE_NAME + "-Data.zip"
