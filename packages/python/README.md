# epistemic-edge

**Air-gapped, neuro-symbolic AIoT framework: 1-bit LLM cognition over mathematically verified, temporally valid edge state.**

## Architecture

Epistemic Edge orchestrates four tiers into a strict **verify-decay-generate** pipeline:

| Tier | Layer | Engine | Function |
|------|-------|--------|----------|
| 1 | Transport | `cbor-ld-ex` | Hyper-compressed binary payloads over MQTT/CoAP |
| 2 | Trust | `jsonld-ex` | Subjective Logic fusion + PROV-O audit trail |
| 3 | Memory | `chronofy` | Temporal-Logical Decay Architecture (TLDA) |
| 4 | Cognition | `llama-cpp-python` | Grammar-constrained 1-bit LLM inference |

The key insight: **1-bit quantized models lose continuous probabilistic nuance**, making them susceptible to hallucination over conflicting or stale context. Epistemic Edge solves this by mathematically guaranteeing the truth of the state graph *before* the LLM ever touches it.

## Installation

```bash
# Core framework (transport + trust + memory)
pip install epistemic-edge

# With local LLM inference
pip install epistemic-edge[llm]

# With MQTT/CoAP transport
pip install epistemic-edge[transport]

# Everything
pip install epistemic-edge[all]
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

## Core Libraries

- **[jsonld-ex](https://github.com/jemsbhai/jsonld-ex)** - JSON-LD 1.2 extensions with Subjective Logic, FHIR R4, PROV-O
- **[cbor-ld-ex](https://github.com/jemsbhai/cbor-ld-ex)** - Compact Binary Linked Data for constrained IoT networks
- **[chronofy](https://github.com/jemsbhai/chronofy)** - Temporal validity framework implementing TLDA

## License

MIT
