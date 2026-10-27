"""Packaging: one version (in ``inksweep.__init__``), recorded in the changelog; the command line answers --version."""
import re
import tomllib
from pathlib import Path

import pytest

import inksweep

ROOT = Path(__file__).resolve().parents[1]


def test_version_has_one_source_and_a_changelog_entry():
    assert re.fullmatch(r"\d+\.\d+\.\d+", inksweep.__version__)
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("not a source checkout")
    project = tomllib.loads(pyproject.read_text())["project"]
    assert "version" not in project and "version" in project["dynamic"]
    changelog = ROOT / "CHANGELOG.md"
    assert f"## {inksweep.__version__}" in changelog.read_text()


def test_version_flag(capsys):
    from inksweep import cli

    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == f"inksweep {inksweep.__version__}"
