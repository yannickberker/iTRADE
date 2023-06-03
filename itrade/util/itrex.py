"""Access the iTReX web app."""
import os
import re
import time
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias, get_args
from urllib.request import urlretrieve

from packaging import version
from selenium.common.exceptions import (
    SessionNotCreatedException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver import Chrome, Firefox, chrome, firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ITREX_URL = "https://itrex.kitz-heidelberg.de/iTReX/"

MET_LAYOUT_FILE = "iTReX-Demo_MRA_Layout.xlsx"
IMA_LAYOUT_FILE = "iTReX-Demo_MRA_Layout-Imaging-3Plates.xlsx"

RESULT_PREFIX = "iTReX-Results"

BrowserType = Literal["chrome", "firefox"]
Version: TypeAlias = version.Version | version.InfiniteTypes


class VersionMismatchErrorFirefox(ValueError):
    """Firefox requires proper MIME type of zip files (iTReX >v1.1.1)."""


def on_exit(itrex_version: Version, wait: int = 120) -> None:
    """Clean up (to be used when we triggered iTReX processing)."""
    if itrex_version <= version.Version("1.1.1"):
        # Wait for iTReX server session to end, and srm cleanup to finish.
        time.sleep(wait)


def download_demo_file(filename: str, download_folder: Path) -> Path:
    """Download file from iTReX."""
    url = ITREX_URL + "/demo/" + filename
    (path, _) = urlretrieve(url, download_folder / filename)  # nosec
    return Path(path)


def make_driver(
    browser: BrowserType | None = None,
    headless: bool = False,
    download_folder: Path = Path(),
) -> Callable[[], WebDriver]:
    """Generate the get_driver function for cohort_download."""
    if browser is None:
        for browser in get_args(BrowserType):
            try_get_driver = make_driver(browser, True, download_folder)
            try:
                with try_get_driver():
                    ...
            except (WebDriverException, SessionNotCreatedException):
                continue
            return try_get_driver
        raise RuntimeError("Could not instantiate any web driver.")

    if browser == "chrome":
        if headless:
            # https://stackoverflow.com/q/50790733#comment91362049_50790733
            os.environ["DISPLAY"] = ""
        cro = chrome.options.Options()
        if headless:
            cro.add_argument("--headless")
        prefs = {
            "download.default_directory": str(download_folder),
            "download.prompt_for_download": False,
        }
        cro.add_experimental_option("prefs", prefs)

        def get_driver() -> WebDriver:
            return Chrome(options=cro)

    elif browser == "firefox":
        ffo = firefox.options.Options()
        if headless:
            ffo.add_argument("--headless")
        ffo.set_preference("browser.download.folderList", 2)
        ffo.set_preference("browser.download.manager.showWhenStarting", False)
        ffo.set_preference("browser.download.dir", str(download_folder))
        mime_types = "application/zip, text/html"
        ffo.set_preference("browser.helperApps.neverAsk.saveToDisk", mime_types)

        def get_driver() -> WebDriver:
            return Firefox(options=ffo)

    return get_driver


def process_cohort(
    layout_file: Path,
    readouts_file: Path,
    download_folder: Path,
    get_driver: Callable[[], WebDriver] | None = None,
    gui_timeout: int = 30,
    pre_download_wait: int = 5,
    download_timeout: int = 3600,
    name: str | None = None,
) -> Path:
    """Process cohort in iTReX and download results."""
    if get_driver is None:
        get_driver = make_driver(headless=True, download_folder=download_folder)

    with get_driver() as driver:
        wait = WebDriverWait(driver, gui_timeout)

        def wait_title(title: str) -> Any:
            return wait.until(EC.title_contains(title))

        def wait_invisible(*args: str) -> Any:
            return wait.until(EC.invisibility_of_element(args))

        def wait_click(*args: str) -> Any:
            return wait.until(EC.element_to_be_clickable(args))

        def wait_and_upload(element: str, path: Path) -> Any:
            driver.find_element(By.ID, element).send_keys(str(path))
            progress_selector = (
                f"#{element}_progress > .progress-bar[style*='100']"
                + ":not(.progress-bar-danger)"
            )
            wait.until(
                EC.visibility_of_any_elements_located(
                    (By.CSS_SELECTOR, progress_selector)
                )
            )

        driver.get(ITREX_URL)
        wait_title("iTReX")

        # Find iTReX version
        itrex_version: Version
        elements = driver.find_elements(By.CSS_SELECTOR, "h4[title]")
        for element in elements:
            title_string = element.get_attribute("title")
            if title_string is None:
                continue

            version_string = re.search(version.VERSION_PATTERN, title_string, re.X)
            if version_string is None:
                continue

            itrex_version = version.parse(version_string[0])
            break
        else:
            itrex_version = version.Infinity

        if itrex_version <= version.Version("1.1.1") and isinstance(driver, Firefox):
            raise VersionMismatchErrorFirefox

        try:
            wait_click(By.ID, "accept").click()
            wait_invisible(By.CLASS_NAME, "modal-backdrop")
        except TimeoutException:
            # We don't see the dialog at all in R/shinytest, so skip it here, too.
            pass

        cohort_selector = "input[name=number_of_samples][value=cohort]"
        wait_click(By.CSS_SELECTOR, cohort_selector).click()
        wait_and_upload("layout_table_file", path=layout_file)
        wait_and_upload("readout_matrices_file", path=readouts_file)

        wait_click(By.ID, "start_oca").click()

        wait = WebDriverWait(driver, download_timeout)
        download_button = wait_click(By.ID, "Archive")

        def get_candidate_files() -> set[Path]:
            return set(
                chain.from_iterable(
                    download_folder.glob(pattern)
                    for pattern in [f"{RESULT_PREFIX}_*.zip", "*.html"]
                )
            )

        files_before = get_candidate_files()

        def get_new_file(files_before: set[Path]) -> Path | None:
            files = get_candidate_files() - files_before
            return files.pop() if files else None

        deadline = time.time() + download_timeout
        while time.time() < deadline:
            time.sleep(pre_download_wait)
            download_button.click()

            while time.time() < deadline and not (file_ := get_new_file(files_before)):
                time.sleep(1)

            if file_:
                # We expect a non-HTML file
                if file_.suffix != ".html":
                    break
                # yet sometimes see download failures in the form of some .html file:
                files_before.add(file_)
        else:
            on_exit(itrex_version)
            raise TimeoutError("Timeout waiting for creation of new non-HTML file.")

        size = -1
        while time.time() < deadline:
            if size == (size := file_.stat().st_size) and size > 0:
                break
            time.sleep(1)
        else:
            on_exit(itrex_version)
            raise TimeoutError("Timeout waiting for constant positive file size.")

        on_exit(itrex_version)

        if name is not None:
            new_name = file_.name.replace(RESULT_PREFIX, f"{RESULT_PREFIX}-{name}")
            file_ = file_.rename(file_.parent / new_name)

        return file_
