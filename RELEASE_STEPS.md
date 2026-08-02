# Release steps — epistemic-edge v0.1.0

PowerShell, from the repo root unless noted. Do not skip steps.

## Phase 1 — repository (pre-publish gate)

1. `cd packages\python`
2. `poetry check --lock` → must exit 0 (legacy `[tool.poetry]` warnings are expected and harmless).
3. `poetry sync --all-extras` → dev venv matches the lock exactly.
4. `poetry run pytest tests/ -v --cov=epistemic_edge --cov-report=term` → all green. No exceptions.
5. Commit + push (explicit approval per commit; conventional messages).
6. Independent verification of the public repo state (fresh clone with
   `--recurse-submodules`, build, test) before Phase 2.

## Phase 2 — publish

```powershell
cd packages\python
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
poetry build

# TestPyPI first
poetry publish -r testpypi

# Verify from TestPyPI in a clean venv.
# --extra-index-url is REQUIRED: dependencies (jsonld-ex, cbor-ld-ex, chronofy, ...)
# live on real PyPI, not TestPyPI.
python -m venv test_env
.\test_env\Scripts\Activate.ps1
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ epistemic-edge==0.1.0
python -c "import epistemic_edge as ee; print('core OK', ee.__version__)"
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "epistemic-edge[all]==0.1.0"
python -c "import aiomqtt, aiocoap, llama_cpp; print('extras OK')"
deactivate
Remove-Item -Recurse -Force test_env

# Only after TestPyPI verification passes — production PyPI
poetry publish

# Verify production
pip install epistemic-edge==0.1.0
```

## Phase 3 — post-publish

```powershell
git tag -a v0.1.0 -m "Release v0.1.0 - IEEE IRI 2026 paper release"
git push origin v0.1.0
```

- Create a GitHub Release for `v0.1.0` using the `[0.1.0]` section of
  `packages/python/CHANGELOG.md` as the notes.
