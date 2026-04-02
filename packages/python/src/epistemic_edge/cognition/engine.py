"""
Cognition engine: wraps llama-cpp-python for 1-bit LLM inference.

Feeds the verified, temporally fresh StateGraph into a local GGUF model
and produces grammar-constrained EdgeIntent objects.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from epistemic_edge.models import EdgeIntent, StateGraph

logger = logging.getLogger(__name__)


class CognitionEngine:
    """
    Local LLM inference engine for edge cognition.

    Designed for 1-bit quantized models (e.g., Bonsai GGUF) but works
    with any llama.cpp-compatible model file.
    """

    def __init__(
        self,
        model_path: str,
        context_size: int = 2048,
        max_tokens: int = 256,
    ) -> None:
        self.model_path = model_path
        self.context_size = context_size
        self.max_tokens = max_tokens
        self._llm: Any = None

    def _load_model(self) -> None:
        """Lazy-load the GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "Cognition engine requires the 'llm' extra: "
                "pip install epistemic-edge[llm]"
            ) from exc

        logger.info("Loading model from %s (ctx=%d)", self.model_path, self.context_size)
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.context_size,
            verbose=False,
        )

    def _build_prompt(self, state: StateGraph, query: str) -> str:
        """
        Construct a prompt from the active state graph and operator query.

        The prompt is kept minimal to respect edge compute constraints.
        """
        facts_block = "\n".join(
            f"- [b={f.belief:.2f} d={f.disbelief:.2f} u={f.uncertainty:.2f}] {f.payload}"
            for f in state.facts
        )

        return (
            f"You are an AIoT edge controller for node '{state.node_id}'.\n"
            f"Active verified state ({state.active_count} facts):\n"
            f"{facts_block}\n\n"
            f"Operator query: {query}\n\n"
            f"Respond with a JSON object containing: "
            f'"action", "target", and optional "parameters".\n'
        )

    async def generate_intent(self, state: StateGraph, query: str) -> EdgeIntent:
        """
        Generate a grammar-constrained EdgeIntent from the current state.

        Runs inference in a thread pool to avoid blocking the async loop.
        """
        if self._llm is None:
            self._load_model()

        prompt = self._build_prompt(state, query)

        # Run blocking llama.cpp inference in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._llm(prompt, max_tokens=self.max_tokens),
        )

        raw_text = result["choices"][0]["text"].strip()
        logger.debug("Raw LLM output: %s", raw_text)

        return self._parse_intent(raw_text)

    def _parse_intent(self, raw: str) -> EdgeIntent:
        """
        Parse the LLM output into an EdgeIntent.

        Attempts JSON parsing first; falls back to a safe no-op intent
        if the output is malformed.
        """
        try:
            data = json.loads(raw)
            return EdgeIntent(
                action=data.get("action", "noop"),
                target=data.get("target", "unknown"),
                parameters=data.get("parameters", {}),
                raw_llm_output=raw,
                grammar_constrained=False,  # TODO: True when GBNF is wired
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse LLM output as JSON: %s", exc)
            return EdgeIntent(
                action="noop",
                target="parse_failure",
                parameters={"raw": raw, "error": str(exc)},
                raw_llm_output=raw,
                grammar_constrained=False,
            )
