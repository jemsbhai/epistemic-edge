# Changelog

All notable changes to the `epistemic-edge` package.
Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-08-02

**IEEE IRI 2026 paper release.** Companion release for *"Epistemic Edge: Subjective
Logic Guardrails for LLM-Driven IoT Actuation"* (Syed & Silaghi, IEEE IRI 2026).

### Added
- `epistemic_edge.analysis.metrics`: pure analysis functions used for the paper's
  evaluation — confusion counts, classification metrics, Cohen's kappa, paired
  Cohen's d — with dedicated unit tests.
- `tests/test_version.py`: release guard asserting `__version__` matches
  `pyproject.toml`.
- Repository-root `README.md` and `LICENSE` (paper reference, BibTeX, quickstart,
  tier-to-module map, reproduction pointers).
- `poetry.lock` is now committed, pinning the development environment that produced
  the paper's results.

### Changed
- Version bumped 0.0.1 → 0.1.0.

### Removed
- `numpy` and `scipy` removed from main dependencies — nothing in the package,
  tests, or experiment runners imports them. (`numpy` may still be installed as a
  transitive dependency of the optional `llama-cpp-python`.)

### Fixed
- Packaging: the `transport`, `llm`, and `all` extras are now declared as optional
  main dependencies and appear in the published wheel metadata. Previously they
  referenced Poetry dependency groups, which are not published, so
  `pip install "epistemic-edge[llm]"` (and the other extras) installed nothing
  beyond the core package.

## [0.0.1] — 2026-04-02

Initial public release: four-tier verify–decay–generate pipeline (transport, trust,
memory, cognition), dataset adapters (BATADAL, SWaT, NASA bearing, MQTT-IoT),
subjective-logic fusion with PROV-O audit trail, grammar-constrained local LLM
inference, typed `src/` layout with full test suite.
