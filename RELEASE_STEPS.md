# Release steps — epistemic-edge v0.1.0

PowerShell, from the repo root unless noted. Do not skip steps.

## Phase 1 — repository (pre-publish gate) — COMPLETED 2026-08-02

1. `poetry check --lock` → exit 0. ✓
2. Hermetic test suite in the project venv (direct interpreter invocation:
   `.venv\Scripts\python.exe -m pytest tests/`) → 202 passed, 75% coverage. ✓
3. Committed and pushed as 7 conventional commits (`17fc8f3..de0aaf6`). ✓
4. Independent verification from a fresh public clone (Claude container):
   `git clone --recurse-submodules` → submodule pinned at `f5dda72` →
   `python -m build` → `twine check dist/*` PASSED → clean-venv wheel install →
   Quick Start API verified signature-by-signature → live `[transport]` extra
   install → 202/202 tests against the installed wheel. ✓

## Phase 2 — publish (direct to PyPI; TestPyPI intentionally skipped)

The Phase 1 gates substitute for TestPyPI: `twine check` covers metadata
acceptance, and the clean-venv wheel install covers install-ability.

**A published version number is permanent** — PyPI never allows re-uploading the
same version, even after deletion. Any post-publish fix must ship as 0.1.1.

```powershell
cd packages\python
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
poetry build

# Metadata gate — must PASS before publishing.
# (If twine is missing from the venv: .venv\Scripts\python.exe -m pip install twine)
.venv\Scripts\python.exe -m twine check dist/*

# Production PyPI — uses your configured Poetry credentials. Run from your own
# terminal. This step is the point of no return for the 0.1.0 version number.
poetry publish

# Post-publish verification in a fresh venv
python -m venv test_env
.\test_env\Scripts\Activate.ps1
pip install epistemic-edge==0.1.0
python -c "import epistemic_edge as ee; print('PyPI install OK', ee.__version__)"
deactivate
Remove-Item -Recurse -Force test_env
```

Environment note: invoke the project venv's interpreter directly
(`.venv\Scripts\python.exe`) rather than relying on `poetry run` / `poetry sync`
from shells that may inherit a foreign `VIRTUAL_ENV`.

## Phase 3 — post-publish

```powershell
git tag -a v0.1.0 -m "Release v0.1.0 - IEEE IRI 2026 paper release"
git push origin v0.1.0
```

- Create a GitHub Release for `v0.1.0` using the `[0.1.0]` section of
  `packages/python/CHANGELOG.md` as the notes.
