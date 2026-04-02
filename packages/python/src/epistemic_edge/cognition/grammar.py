"""
GBNF grammar compiler for constrained LLM decoding.

Compiles jsonld-ex compliance schemas into Generative Backus-Naur Form
(GBNF) grammars that physically restrict the LLM from outputting anything
other than valid JSON-LD actuation intents.

This is the key safety mechanism: a 1-bit model cannot hallucinate
invalid commands when its token generation is grammar-constrained.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Minimal GBNF grammar for EdgeIntent JSON output
EDGE_INTENT_GRAMMAR = r"""
root   ::= "{" ws "\"action\"" ws ":" ws string "," ws "\"target\"" ws ":" ws string ( "," ws "\"parameters\"" ws ":" ws object )? ws "}"
string ::= "\"" [a-zA-Z0-9_\-/.]+ "\""
object ::= "{" ws ( string ws ":" ws value ( "," ws string ws ":" ws value )* )? ws "}"
value  ::= string | number | "true" | "false" | "null"
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws     ::= [ \t\n]*
""".strip()


class GBNFCompiler:
    """
    Compiles structured schemas into GBNF grammars for llama.cpp.

    Can produce grammars from:
    - A fixed set of allowed actions (action whitelist)
    - A jsonld-ex compliance schema (planned)
    - Custom rules
    """

    def __init__(self) -> None:
        self._allowed_actions: list[str] | None = None

    def with_allowed_actions(self, actions: list[str]) -> GBNFCompiler:
        """Restrict the grammar to only permit specific action strings."""
        self._allowed_actions = actions
        return self

    def compile(self) -> str:
        """
        Produce the GBNF grammar string.

        If allowed_actions are set, the action field is restricted
        to those specific values. Otherwise, any alphanumeric string is valid.
        """
        if self._allowed_actions:
            return self._compile_restricted()
        return EDGE_INTENT_GRAMMAR

    def _compile_restricted(self) -> str:
        """Build a grammar where the action field is an enum."""
        if not self._allowed_actions:
            return EDGE_INTENT_GRAMMAR

        alternatives = " | ".join(
            f'"\\"{a}\\""' for a in self._allowed_actions
        )
        action_rule = f"action ::= {alternatives}"

        return (
            f'root   ::= "{{" ws "\\"action\\"" ws ":" ws action "," '
            f'ws "\\"target\\"" ws ":" ws string '
            f'( "," ws "\\"parameters\\"" ws ":" ws object )? ws "}}"}\n'
            f"{action_rule}\n"
            f'string ::= "\\"" [a-zA-Z0-9_\\-/.]+ "\\""\n'
            f'object ::= "{{" ws ( string ws ":" ws value '
            f'( "," ws string ws ":" ws value )* )? ws "}}"\n'
            f'value  ::= string | number | "true" | "false" | "null"\n'
            f'number ::= "-"? [0-9]+ ("." [0-9]+)?\n'
            f'ws     ::= [ \\t\\n]*'
        )

    @staticmethod
    def from_jsonld_schema(schema: dict[str, Any]) -> GBNFCompiler:
        """
        Compile a GBNF grammar from a jsonld-ex compliance schema.

        This extracts the allowed action types and parameter shapes
        from the schema and builds a grammar that enforces them at
        the token level during LLM decoding.

        TODO: Full implementation pending jsonld-ex schema introspection API.
        """
        compiler = GBNFCompiler()
        actions = schema.get("allowed_actions", [])
        if actions:
            compiler.with_allowed_actions(actions)
        logger.info("Compiled GBNF from schema with %d allowed actions.", len(actions))
        return compiler
