"""Comprehensive tests for cognition/engine.py — CognitionEngine."""

import json

import pytest

from epistemic_edge.cognition.engine import CognitionEngine
from epistemic_edge.models import EdgeIntent, FusedState, StateGraph


class TestBuildPrompt:
    """Tests for CognitionEngine._build_prompt (no model required)."""

    @pytest.fixture
    def engine(self) -> CognitionEngine:
        return CognitionEngine(model_path="/fake/model.gguf")

    def test_includes_node_id(self, engine: CognitionEngine) -> None:
        state = StateGraph(node_id="gateway_42")
        prompt = engine._build_prompt(state, "status?")
        assert "gateway_42" in prompt

    def test_includes_query(self, engine: CognitionEngine) -> None:
        state = StateGraph(node_id="n1")
        prompt = engine._build_prompt(state, "What is the pipeline pressure?")
        assert "What is the pipeline pressure?" in prompt

    def test_includes_fact_count(self, engine: CognitionEngine) -> None:
        state = StateGraph(
            node_id="n1",
            facts=[
                FusedState(payload={"a": 1}, belief=0.9, disbelief=0.05, uncertainty=0.05),
                FusedState(payload={"b": 2}, belief=0.7, disbelief=0.1, uncertainty=0.2),
            ],
        )
        prompt = engine._build_prompt(state, "report")
        assert "2 facts" in prompt

    def test_includes_sl_bounds_in_facts(self, engine: CognitionEngine) -> None:
        state = StateGraph(
            node_id="n1",
            facts=[
                FusedState(payload={"temp": 55}, belief=0.85, disbelief=0.10, uncertainty=0.05),
            ],
        )
        prompt = engine._build_prompt(state, "q")
        assert "b=0.85" in prompt
        assert "d=0.10" in prompt
        assert "u=0.05" in prompt

    def test_empty_state_produces_valid_prompt(self, engine: CognitionEngine) -> None:
        state = StateGraph(node_id="empty_node")
        prompt = engine._build_prompt(state, "anything happening?")
        assert "0 facts" in prompt
        assert "empty_node" in prompt

    def test_requests_json_output(self, engine: CognitionEngine) -> None:
        state = StateGraph(node_id="n1")
        prompt = engine._build_prompt(state, "q")
        assert "JSON" in prompt
        assert '"action"' in prompt
        assert '"target"' in prompt


class TestParseIntent:
    """Tests for CognitionEngine._parse_intent (no model required)."""

    @pytest.fixture
    def engine(self) -> CognitionEngine:
        return CognitionEngine(model_path="/fake/model.gguf")

    def test_valid_json(self, engine: CognitionEngine) -> None:
        raw = json.dumps({"action": "close_valve", "target": "valve_3"})
        intent = engine._parse_intent(raw)
        assert intent.action == "close_valve"
        assert intent.target == "valve_3"
        assert intent.raw_llm_output == raw

    def test_valid_json_with_parameters(self, engine: CognitionEngine) -> None:
        raw = json.dumps({
            "action": "set_speed",
            "target": "motor_1",
            "parameters": {"rpm": 1500, "ramp_time": 10},
        })
        intent = engine._parse_intent(raw)
        assert intent.action == "set_speed"
        assert intent.parameters["rpm"] == 1500

    def test_invalid_json_returns_noop(self, engine: CognitionEngine) -> None:
        raw = "This is not JSON at all"
        intent = engine._parse_intent(raw)
        assert intent.action == "noop"
        assert intent.target == "parse_failure"
        assert "raw" in intent.parameters
        assert intent.raw_llm_output == raw

    def test_partial_json_missing_action(self, engine: CognitionEngine) -> None:
        raw = json.dumps({"target": "valve_1"})
        intent = engine._parse_intent(raw)
        assert intent.action == "noop"  # Falls back to default
        assert intent.target == "valve_1"

    def test_partial_json_missing_target(self, engine: CognitionEngine) -> None:
        raw = json.dumps({"action": "stop"})
        intent = engine._parse_intent(raw)
        assert intent.action == "stop"
        assert intent.target == "unknown"  # Falls back to default

    def test_empty_string_returns_noop(self, engine: CognitionEngine) -> None:
        intent = engine._parse_intent("")
        assert intent.action == "noop"
        assert intent.target == "parse_failure"

    def test_json_with_extra_fields_ignored(self, engine: CognitionEngine) -> None:
        raw = json.dumps({
            "action": "open",
            "target": "door",
            "explanation": "Operator requested entry",
            "confidence": 0.95,
        })
        intent = engine._parse_intent(raw)
        assert intent.action == "open"
        assert intent.target == "door"
        # Extra fields don't cause errors

    def test_grammar_constrained_flag_false(self, engine: CognitionEngine) -> None:
        raw = json.dumps({"action": "x", "target": "y"})
        intent = engine._parse_intent(raw)
        assert intent.grammar_constrained is False


class TestCognitionEngineInit:
    """Tests for CognitionEngine construction."""

    def test_defaults(self) -> None:
        engine = CognitionEngine(model_path="/path/to/model.gguf")
        assert engine.model_path == "/path/to/model.gguf"
        assert engine.context_size == 2048
        assert engine.max_tokens == 256
        assert engine._llm is None

    def test_custom_context(self) -> None:
        engine = CognitionEngine(
            model_path="/m.gguf", context_size=4096, max_tokens=512
        )
        assert engine.context_size == 4096
        assert engine.max_tokens == 512

    def test_load_model_without_llama_cpp_raises(self) -> None:
        """If llama-cpp-python is not installed, _load_model should give a clear error."""
        engine = CognitionEngine(model_path="/nonexistent.gguf")
        # This may or may not raise depending on whether llama-cpp-python is installed
        # If installed, it will fail on the nonexistent file path
        # Either way, we're testing it doesn't silently pass
        with pytest.raises(Exception):
            engine._load_model()
