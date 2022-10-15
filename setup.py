"""Package setup."""
import re
from pathlib import Path
from typing import overload

from setuptools import find_packages, setup


def _read_file(basename: str, /, rm_comments: bool = False) -> str:
    text = (Path(__file__).parent / basename).read_text()
    if rm_comments:
        text = re.sub(r"\s+#\s+.*(\n|$)", r"\1", text)
    return text


def _read_requirements(basename: str) -> list[str]:
    return [req for req in _read_file(basename).splitlines() if "==" not in req]


@overload
def _get_yaml_values(yaml: str, tag: str) -> str:
    ...


@overload
def _get_yaml_values(yaml: str, tag: str, is_array_tag: bool) -> list[str]:
    ...


def _get_yaml_values(
    yaml: str, tag: str, is_array_tag: bool = False
) -> str | list[str]:
    sep = "\n  - " if is_array_tag else " "
    match = re.search(f"(?<=^{tag}:{sep})(({sep})?.*)+$", yaml, flags=re.M)
    if match is None:
        raise ValueError(f"Cannot find YAML value for tag {tag}")
    return match[0].split(sep) if is_array_tag else match[0]


cff = _read_file("CITATION.cff", rm_comments=True)
version = _get_yaml_values(cff, "version")
abstract = _get_yaml_values(cff, "abstract")
paper_doi = _get_yaml_values(cff, "url")
github_url = _get_yaml_values(cff, "repository-code")
license_ = _get_yaml_values(cff, "license")
keywords = _get_yaml_values(cff, "keywords", is_array_tag=True)

readme = _read_file("README.md")

requirements = _read_requirements("requirements.txt")
requirements_r = _read_requirements("requirements-R.txt")
requirements_dev = _read_requirements("requirements-dev.txt")

setup(
    name="itrade",
    version=version,
    description=abstract,
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Yannick Berker",
    url=paper_doi,
    download_url=github_url,
    packages=find_packages(),
    package_data={"": ["util/plots.R"]},
    include_package_data=True,
    license=license_,
    keywords=keywords,
    project_urls={
        "Publication": paper_doi,
        "Source": github_url,
        "Tracker": github_url + "/issues",
    },
    install_requires=requirements,
    python_requires=">=3.10",
    setup_requires=["wheel"],
    extras_require={
        "R": requirements_r,
        "dev": requirements_dev,
    },
    entry_points={
        "console_scripts": [
            "itrade-run-cnn=itrade.run_cnn:run_cnn",
            "itrade-run-cnns=itrade.run_all:run_cnns",
            "itrade-run-itrex=itrade.run_all:run_itrex",
            "itrade-run-plots=itrade.run_all:run_plots",
            "itrade-run-all=itrade.run_all:run_all",
            "itrade-test-gpu=itrade.test_gpu:test_gpu",
        ],
    },
)
