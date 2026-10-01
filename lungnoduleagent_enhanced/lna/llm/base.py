"""Abstract LLM backend.

All language-model access in the system goes through this interface, so the
same pipeline runs against the deterministic mock (default), Anthropic, or
OpenAI without changing any agent code.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class LLMBackend(abc.ABC):
    @abc.abstractmethod
    def complete(self, system: str, prompt: str,
                 context: Optional[Dict[str, Any]] = None) -> str:
        """Free-text completion."""

    @abc.abstractmethod
    def structured(self, system: str, prompt: str, task: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a JSON-like dict. `task` names the expected schema so the
        mock backend knows what to synthesize and real backends can attach the
        right response format / tool schema."""
