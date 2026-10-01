"""CountingBackend — wraps any LLMBackend and tallies calls.

Every ``complete`` / ``structured`` call is counted, bucketed by the ``task``
tag in the context. This is what turns "cost-aware" from a claim into a
measurement: the pipeline reads ``backend.calls`` per case and the evaluator
compares adaptive routing against an always-full-board reference.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Optional

from .base import LLMBackend


#: characters per token, the standard rough conversion for English prompts.
CHARS_PER_TOKEN = 4.0


class CountingBackend(LLMBackend):
    def __init__(self, inner: LLMBackend):
        self.inner = inner
        self.calls: Counter = Counter()
        self.total: int = 0
        self.in_chars: int = 0
        self.out_chars: int = 0

    def reset(self) -> None:
        self.calls.clear()
        self.total = 0
        self.in_chars = 0
        self.out_chars = 0

    def snapshot(self) -> Dict[str, int]:
        return dict(self.calls)

    @property
    def prompt_tokens(self) -> float:
        """Approximate prompt tokens sent. The prompts are the agents' own, so
        this is a property of the architecture, not of the backend behind it."""
        return self.in_chars / CHARS_PER_TOKEN

    @property
    def total_tokens(self) -> float:
        return (self.in_chars + self.out_chars) / CHARS_PER_TOKEN

    def _bump(self, task: str) -> None:
        self.calls[task] += 1
        self.total += 1

    def complete(self, system: str, prompt: str,
                 context: Optional[Dict[str, Any]] = None) -> str:
        self._bump((context or {}).get("task", "complete"))
        self.in_chars += len(system or "") + len(prompt or "")
        out = self.inner.complete(system, prompt, context)
        self.out_chars += len(out or "")
        return out

    def structured(self, system: str, prompt: str, task: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._bump(task)
        self.in_chars += len(system or "") + len(prompt or "")
        out = self.inner.structured(system, prompt, task, context)
        self.out_chars += len(json.dumps(out, default=str))
        return out
