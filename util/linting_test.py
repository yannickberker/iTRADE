"""Test pylint settings."""
import io
from contextlib import redirect_stdout
from pathlib import Path

import pylint.lint


def test_useless_suppression(tmp_path: Path) -> None:
    """Test that pylint issues useless-suppression warnings."""
    tmp_file = tmp_path / "pylint_useless_supression.py"
    tmp_file.write_text("# pylint:disable=no-member\n")

    with redirect_stdout(io.StringIO()) as stdout:
        pylint.lint.Run([str(tmp_file)], exit=False)

    assert "(useless-suppression)" in stdout.getvalue()
