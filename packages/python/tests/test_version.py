"""Release guard: the package version must stay consistent across metadata.

Catches the classic release drift where ``pyproject.toml`` is bumped but
``epistemic_edge.__version__`` is not (or vice versa).
"""

import re
from pathlib import Path

import pytest

import epistemic_edge


def _pyproject_version() -> str:
    tomllib = pytest.importorskip("tomllib")  # stdlib on Python >= 3.11
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["tool"]["poetry"]["version"]


def test_dunder_version_matches_pyproject() -> None:
    assert epistemic_edge.__version__ == _pyproject_version()


def test_version_is_semver() -> None:
    assert re.fullmatch(
        r"\d+\.\d+\.\d+(?:[abc]\d+|\.dev\d+|\.post\d+)?",
        epistemic_edge.__version__,
    ), f"__version__ {epistemic_edge.__version__!r} is not a release-style version"
