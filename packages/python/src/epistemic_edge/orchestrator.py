"""
EdgeNode orchestrator: the main entry point for the Epistemic Edge pipeline.

Wires together transport (cbor-ld-ex), trust (jsonld-ex), memory (chronofy),
and cognition (1-bit LLM) into an async event loop that enforces a strict
verify-decay-generate cycle.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Awaitable

from epistemic_edge.models import (
    ActuationResult,
    EdgeIntent,
    FusedState,
    Observation,
    StateGraph,
)
from epistemic_edge.trust.fusion import SLFusion
from epistemic_edge.trust.audit import PROVOAudit
from epistemic_edge.memory.cache import TemporalCache, DecayConfig

logger = logging.getLogger(__name__)


class EdgeNode:
    """
    An autonomous AIoT node running the Epistemic Edge pipeline.

    Usage::

        node = EdgeNode(
            node_id="gateway_alpha",
            decay=DecayConfig(mean_reversion_rate=1.5, threshold=0.2),
        )

        @node.guardrail(action="close_valve")
        def check_safety(state, intent):
            return state.max_uncertainty() < 0.15

        @node.on_actuate
        async def execute(intent, receipt):
            ...

        await node.start()
    """

    def __init__(
        self,
        node_id: str,
        decay: DecayConfig | None = None,
        llm_path: str | None = None,
        llm_context_size: int = 2048,
        audit_mode: str = "prov-o",
    ) -> None:
        self.node_id = node_id
        self.state = StateGraph(node_id=node_id)
        self.decay_config = decay or DecayConfig()
        self.llm_path = llm_path
        self.llm_context_size = llm_context_size
        self.audit_mode = audit_mode

        # Internal components (lazy-initialized)
        self._fusion = SLFusion()
        self._audit = PROVOAudit() if audit_mode == "prov-o" else None
        self._cache = TemporalCache(config=self.decay_config)
        self._engine: Any = None  # Cognition engine, lazy

        # User-registered callbacks
        self._guardrails: dict[str, Callable[..., bool]] = {}
        self._actuate_handler: Callable[..., Awaitable[None]] | None = None

        self._running = False

    # -- Decorator API --------------------------------------------------------

    def guardrail(self, action: str) -> Callable[..., Callable[..., bool]]:
        """Register a guardrail check for a specific action type."""
        def decorator(fn: Callable[..., bool]) -> Callable[..., bool]:
            self._guardrails[action] = fn
            return fn
        return decorator

    def on_actuate(self, fn: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        """Register the physical actuation handler."""
        self._actuate_handler = fn
        return fn

    # -- Pipeline stages ------------------------------------------------------

    def ingest(self, observation: Observation) -> FusedState:
        """
        Tier 1 + 2: Accept a raw observation, fuse it into the state graph
        via jsonld-ex Subjective Logic, and log the PROV-O activity.
        """
        fused = self._fusion.fuse_observation(observation, self.state)
        if self._audit:
            self._audit.log_fusion(observation, fused)
        self.state.facts.append(fused)
        return fused

    def sweep(self, query_time: datetime | None = None) -> int:
        """
        Tier 3: Run chronofy temporal decay over the state graph.

        Returns the number of facts pruned.
        """
        now = query_time or datetime.utcnow()
        fresh, stale = self._cache.partition(self.state.facts, now)
        pruned = len(self.state.facts) - len(fresh)
        self.state.facts = fresh
        self.state.stale.extend(stale)
        self.state.last_sweep = now
        return pruned

    async def generate(self, query: str) -> EdgeIntent | None:
        """
        Tier 4: Feed the verified, temporally fresh state into the 1-bit LLM
        and produce a grammar-constrained EdgeIntent.

        Returns None if no LLM is configured.
        """
        if self.llm_path is None:
            logger.warning("No LLM configured; skipping cognition tier.")
            return None

        # Lazy import to keep the package light when llm extra is not installed
        from epistemic_edge.cognition.engine import CognitionEngine

        if self._engine is None:
            self._engine = CognitionEngine(
                model_path=self.llm_path,
                context_size=self.llm_context_size,
            )

        return await self._engine.generate_intent(self.state, query)

    def verify_intent(self, intent: EdgeIntent) -> ActuationResult:
        """
        Run the registered guardrail for the intent's action type.

        If no guardrail is registered for the action, the intent is denied
        by default (fail-closed).
        """
        check = self._guardrails.get(intent.action)
        if check is None:
            return ActuationResult(
                intent=intent,
                permitted=False,
                reason=f"No guardrail registered for action '{intent.action}'",
            )

        try:
            ok = check(self.state, intent)
        except Exception as exc:
            return ActuationResult(
                intent=intent,
                permitted=False,
                reason=f"Guardrail raised: {exc}",
            )

        result = ActuationResult(
            intent=intent,
            permitted=ok,
            reason="" if ok else "Guardrail check returned False",
            executed_at=datetime.utcnow() if ok else None,
        )
        if self._audit and ok:
            result.prov_o_receipt_id = self._audit.log_actuation(intent)
        return result

    # -- Event loop -----------------------------------------------------------

    async def start(self) -> None:
        """Start the background sweep loop. Blocks until stop() is called."""
        self._running = True
        logger.info("EdgeNode '%s' started.", self.node_id)
        try:
            while self._running:
                self.sweep()
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("EdgeNode '%s' cancelled.", self.node_id)
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the event loop to stop."""
        self._running = False
