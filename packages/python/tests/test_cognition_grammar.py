"""Comprehensive tests for cognition/grammar.py — GBNFCompiler."""

import pytest

from epistemic_edge.cognition.grammar import EDGE_INTENT_GRAMMAR, GBNFCompiler


class TestEdgeIntentGrammar:
    """Tests for the default GBNF grammar constant."""

    def test_grammar_is_nonempty_string(self) -> None:
        assert isinstance(EDGE_INTENT_GRAMMAR, str)
        assert len(EDGE_INTENT_GRAMMAR) > 50

    def test_grammar_has_root_rule(self) -> None:
        assert "root" in EDGE_INTENT_GRAMMAR

    def test_grammar_requires_action_field(self) -> None:
        assert "action" in EDGE_INTENT_GRAMMAR

    def test_grammar_requires_target_field(self) -> None:
        assert "target" in EDGE_INTENT_GRAMMAR

    def test_grammar_has_string_rule(self) -> None:
        assert "string" in EDGE_INTENT_GRAMMAR

    def test_grammar_supports_parameters(self) -> None:
        assert "parameters" in EDGE_INTENT_GRAMMAR

    def test_grammar_has_value_types(self) -> None:
        assert "true" in EDGE_INTENT_GRAMMAR
        assert "false" in EDGE_INTENT_GRAMMAR
        assert "null" in EDGE_INTENT_GRAMMAR
        assert "number" in EDGE_INTENT_GRAMMAR


class TestGBNFCompiler:
    """Tests for the GBNFCompiler class."""

    def test_default_compile_returns_base_grammar(self) -> None:
        compiler = GBNFCompiler()
        result = compiler.compile()
        assert result == EDGE_INTENT_GRAMMAR

    def test_with_allowed_actions_returns_self(self) -> None:
        compiler = GBNFCompiler()
        returned = compiler.with_allowed_actions(["open", "close"])
        assert returned is compiler

    def test_restricted_grammar_contains_actions(self) -> None:
        compiler = GBNFCompiler().with_allowed_actions(["close_valve", "open_valve"])
        grammar = compiler.compile()
        assert "close_valve" in grammar
        assert "open_valve" in grammar
        assert "action" in grammar

    def test_restricted_grammar_still_has_structure(self) -> None:
        compiler = GBNFCompiler().with_allowed_actions(["stop"])
        grammar = compiler.compile()
        assert "root" in grammar
        assert "string" in grammar
        assert "ws" in grammar

    def test_single_action(self) -> None:
        compiler = GBNFCompiler().with_allowed_actions(["emergency_stop"])
        grammar = compiler.compile()
        assert "emergency_stop" in grammar

    def test_many_actions(self) -> None:
        actions = [f"action_{i}" for i in range(20)]
        compiler = GBNFCompiler().with_allowed_actions(actions)
        grammar = compiler.compile()
        for a in actions:
            assert a in grammar

    def test_from_jsonld_schema_with_actions(self) -> None:
        schema = {"allowed_actions": ["read_sensor", "calibrate"]}
        compiler = GBNFCompiler.from_jsonld_schema(schema)
        grammar = compiler.compile()
        assert "read_sensor" in grammar
        assert "calibrate" in grammar

    def test_from_jsonld_schema_empty(self) -> None:
        schema = {}
        compiler = GBNFCompiler.from_jsonld_schema(schema)
        grammar = compiler.compile()
        # Falls back to base grammar
        assert grammar == EDGE_INTENT_GRAMMAR

    def test_from_jsonld_schema_no_actions_key(self) -> None:
        schema = {"parameters": {"type": "object"}}
        compiler = GBNFCompiler.from_jsonld_schema(schema)
        grammar = compiler.compile()
        assert grammar == EDGE_INTENT_GRAMMAR

    def test_compile_idempotent(self) -> None:
        compiler = GBNFCompiler().with_allowed_actions(["x", "y"])
        g1 = compiler.compile()
        g2 = compiler.compile()
        assert g1 == g2
