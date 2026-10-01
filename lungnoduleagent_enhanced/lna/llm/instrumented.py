"""CountingBackend — wraps any LLMBackend and tallies calls.

Every ``complete`` / ``structured`` call is counted, bucketed by the ``task``
tag in the context. This is what turns "cost-aware" from a claim into a
measurement: the pipeline reads ``backend.calls`` per case and the evaluator
compares adaptive routing against an always-full-board reference.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from .base import LLMBackend


class CountingBackend(LLMBackend):
    def __init__(self, inner: LLMBackend):
        self.inner = inner
        self.calls: Counter = Counter()
        self.total: int = 0

    def reset(self) -> None:
        self.calls.clear()
        self.total = 0

    def snapshot(self) -> Dict[str, int]:
        return dict(self.calls)

    def _bump(self, task: str) -> None:
        self.calls[task] += 1
        self.total += 1

    def complete(self, system: str, prompt: str,
                 context: Optional[Dict[str, Any]] = None) -> str:
        self._bump((context or {}).get("task", "complete"))
        return self.inner.complete(system, prompt, context)

    def structured(self, system: str, prompt: str, task: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._bump(task)
        return self.inner.structured(system, prompt, task, context)
