# epistemic-edge

**Subjective-logic guardrails for LLM-driven IoT actuation** — an air-gapped,
neuro-symbolic AIoT framework that places calibrated uncertainty quantification
between your sensors and any LLM that can touch the physical world.

Companion package for:

> **Epistemic Edge: Subjective Logic Guardrails for LLM-Driven IoT Actuation.**
> Muntaser Syed and Marius Silaghi. *IEEE 27th International Conference on
> Information Reuse and Integration for Data Science (IEEE IRI 2026)*,
> Seattle, WA. To appear.

Full reproduction materials — experiment scripts, immutable result sets, and the
pinned inference fork — live in the repository:
https://github.com/jemsbhai/epistemic-edge

## Architecture

Epistemic Edge orchestrates four tiers into a strict **verify–decay–generate**
pipeline:

| Tier | Layer | Engine | Function |
|------|-------|--------|----------|
| 1 | Transport | `cbor-ld-ex` | Hyper-compressed binary payloads over MQTT/CoAP |
| 2 | Trust | `jsonld-ex` | Subjective Logic fusion + PROV-O audit trail |
| 3 | Memory | `chronofy` | Temporal decay toward the vacuous opinion (0, 0, 1) |
| 4 | Cognition | `llama-cpp-python` | Grammar-constrained local LLM inference behind threshold + whitelist guardrails |

Locally deployed LLMs can act on conflicting or stale context. Epistemic Edge
annotates every source with a calibrated Subjective Logic opinion, fuses them via
Jøsang cumulative fusion, decays stale beliefs toward uncertainty, and verifies
epistemic thresholds before any actuation is allowed. In the paper's ablation
(seven locally-deployed LLMs, 2,800 controlled trials), removing this epistemic
layer collapses threshold-guardrail accuracy from **1.00 to 0.40 for every
model**. On the BATADAL water-distribution benchmark, calibrated SL opinions
reach **AUROC 0.9004 with no trained weights** and no labeled attack data.

## Installation

```bash
pip install epistemic-edge                # core: transport + trust + memory
pip install "epistemic-edge[llm]"         # + local LLM inference (llama-cpp-python)
pip install "epistemic-edge[transport]"   # + MQTT / CoAP transports
pip install "epistemic-edge[all]"         # everything
```

## Quick Start

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

## Citing

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

## Core Libraries

- **[jsonld-ex](https://github.com/jemsbhai/jsonld-ex)** — JSON-LD 1.2 extensions with Subjective Logic, FHIR R4, PROV-O
- **[cbor-ld-ex](https://github.com/jemsbhai/cbor-ld-ex)** — Compact Binary Linked Data for constrained IoT networks
- **[chronofy](https://github.com/jemsbhai/chronofy)** — Temporal validity framework implementing TLDA

## License

MIT
