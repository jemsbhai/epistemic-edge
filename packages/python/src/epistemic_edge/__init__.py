"""
Epistemic Edge: Air-gapped neuro-symbolic AIoT framework.

Orchestrates cbor-ld-ex (transport), jsonld-ex (trust/fusion),
chronofy (temporal validity), and 1-bit LLM inference into a
unified verify-decay-generate pipeline for constrained edge nodes.
"""

__version__ = "0.0.1"

from epistemic_edge.models import StateGraph, EdgeIntent, ActuationResult
from epistemic_edge.orchestrator import EdgeNode

__all__ = [
    "EdgeNode",
    "StateGraph",
    "EdgeIntent",
    "ActuationResult",
]
