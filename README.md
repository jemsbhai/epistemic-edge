# Epistemic Edge

**Subjective-logic guardrails for LLM-driven IoT actuation** — an air-gapped, neuro-symbolic
AIoT framework that puts calibrated uncertainty quantification between your sensors and any
LLM that can touch the physical world.

[![PyPI](https://img.shields.io/pypi/v/epistemic-edge)](https://pypi.org/project/epistemic-edge/)
[![Python](https://img.shields.io/pypi/pyversions/epistemic-edge)](https://pypi.org/project/epistemic-edge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## The paper

This repository accompanies:

> **Epistemic Edge: Subjective Logic Guardrails for LLM-Driven IoT Actuation.**
> Muntaser Syed and Marius Silaghi.
> *IEEE 27th International Conference on Information Reuse and Integration for Data Science
> (IEEE IRI 2026)*, Seattle, WA. To appear.

Three findings from the paper:

1. **Epistemic guardrails are necessary.** Across seven locally-deployed LLMs (1-bit
   PrismML Bonsai, 4-bit GGUF, and a reasoning model; 2,800 controlled trials), removing
   the epistemic layer collapses threshold-guardrail accuracy from **1.00 to exactly 0.40**
   for every model.
2. **Unsupervised real-data detection.** On the BATADAL water-distribution attack benchmark
   (4,177 h of telemetry, 5 attacks), calibrated SL opinions reach **AUROC 0.9004 with no
   trained weights** — inside the top competition tier, without labeled attack data.
   Unsupervised calibration-quantile thresholds land within 0.04 F1 of the label-using
   optimum.
3. **SL is a decomposable triple.** Belief, disbelief, and uncertainty detect structurally
   different attack classes; no single SL-derived signal wins on all five BATADAL events.

```bibtex
@inproceedings{syed2026epistemic,
  author    = {Syed, Muntaser and Silaghi, Marius},
  title     = {Epistemic Edge: Subjective Logic Guardrails for {LLM}-Driven {IoT} Actuation},
  booktitle = {2026 IEEE 27th International Conference on Information Reuse and
               Integration for Data Science (IRI)},
  year      = {2026},
  note      = {To appear}
}
```

## Installation

```bash
pip install epistemic-edge              # core: transport + trust + memory
pip install "epistemic-edge[llm]"       # + local 1-bit LLM inference (llama-cpp-python)
pip install "epistemic-edge[transport]" # + MQTT / CoAP transports
pip install "epistemic-edge[all]"       # everything
```

## Quick start

```python
import asyncio
from epistemic_edge import EdgeNode
from epistemic_edge.memory import DecayConfig

async def main():
    node = EdgeNode(
        node_id="gateway_alpha",
        llm_path="./models/bonsai-8b-1bit.gguf",
        decay=DecayConfig(mean_reversion_rate=1.5, threshold=0.2),
    )

    @node.guardrail(action="close_valve")
    def check_safety(state, intent):
        return state.max_uncertainty() < 0.15

    @node.on_actuate
    async def execute(intent, receipt):
        print(f"Executing: {intent.action} on {intent.target}")
        print(f"Audit trail: {receipt}")

    await node.start()

asyncio.run(main())
```

## How the package maps to the paper's four tiers

| Paper tier | Function | Package modules |
|---|---|---|
| Tier 1 — Ingest | Per-source SL annotation via calibrated strategies | `epistemic_edge.adapters` |
| Tier 2 — SL Fusion | Jøsang cumulative fusion into a joint epistemic state | `epistemic_edge.trust.fusion` |
| Tier 3 — Decay | Temporal decay toward the vacuous opinion (0, 0, 1) | `epistemic_edge.memory` |
| Tier 4 — LLM + guardrails | Grammar-constrained local inference, threshold + whitelist verification | `epistemic_edge.cognition`, `epistemic_edge.orchestrator` |

## Repository layout

```
packages/python/      the epistemic-edge Python package (src layout, typed, tested)
experiments/          experiment scripts for the paper's evaluations
experiments/results/  released result JSONs (immutable ground truth for reported numbers)
models/               local model assets (downloaded by experiment scripts; never tracked)
prismml-llama.cpp     llama.cpp fork for PrismML Bonsai 1-bit models (git submodule, pinned)
```

## Reproducing the paper's results

```bash
git clone --recurse-submodules https://github.com/jemsbhai/epistemic-edge.git
```

The `prismml-llama.cpp` submodule is pinned to the exact fork commit used for every
reported run. Every number in the paper traces to a JSON in `experiments/results/`
(immutable ground truth). The experiment scripts in `experiments/` regenerate them;
see the per-script headers for dataset prerequisites (BATADAL datasets 03/04 are
distributed by the BATADAL organizers).

## License

MIT — see [LICENSE](LICENSE).

## Contact

Muntaser Syed — msyed2011@my.fit.edu · Marius Silaghi — msilaghi@fit.edu
Florida Institute of Technology
