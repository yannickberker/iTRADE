"""Test iTReX."""
from pathlib import Path

from ..util import itrex

TEST_LAYOUT_FILE_NAME = "iTReX-Demo_MRA_Layout-Imaging-1Plate_ST06.xlsx"
TEST_READOUTS_FILE_NAME = "iTReX-Demo_MRA_Readout-Imaging_BT-40-V3-DS1_ST05.xlsx"


def process_demo_cohort(browser: itrex.BrowserType | None, tmp_path: Path) -> None:
    """Test download and processing of demo data."""
    if browser is None:
        get_driver = None
    else:
        get_driver = itrex.make_driver(
            browser=browser, headless=True, download_folder=tmp_path
        )

    layouts = itrex.download_demo_file(TEST_LAYOUT_FILE_NAME, tmp_path)
    readouts = itrex.download_demo_file(TEST_READOUTS_FILE_NAME, tmp_path)

    zip_file = itrex.process_cohort(
        layouts, readouts, tmp_path, get_driver=get_driver, download_timeout=120
    )
    assert zip_file.is_file()


def test_default(tmp_path: Path) -> None:
    """Test the first in a sequence of possible browsers."""
    process_demo_cohort(None, tmp_path)


def test_chrome(tmp_path: Path) -> None:
    """Test Chrome - requires chromedriver to be installed."""
    process_demo_cohort("chrome", tmp_path)


def test_firefox(tmp_path: Path) -> None:
    """Test Firefox - requires geckodriver to be installed."""
    try:
        process_demo_cohort("firefox", tmp_path)
    except itrex.VersionMismatchErrorFirefox:
        pass
