"""Optional real backend using the OpenAI API.

Only imported if config.backend == 'openai'. Requires:
    pip install openai
    export OPENAI_API_KEY=...
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .anthropic_backend import _SCHEMhints, _extract_json
from .base import LLMBackend


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o"):
        try:
            import openai  # noqa
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai to use the openai backend") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI
        self._client = OpenAI()
        self.model = model

    def _call(self, system: str, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def complete(self, system: str, prompt: str,
                 context: Optional[Dict[str, Any]] = None) -> str:
        return self._call(system, prompt)

    def structured(self, system: str, prompt: str, task: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        hint = _SCHEMhints.get(task, "{}")
        full = f"{prompt}\n\nRespond with ONLY valid JSON matching:\n{hint}"
        raw = self._call(system + " You always answer with strict JSON.", full)
        return _extract_json(raw)
