"""Optional real backend using the Anthropic API.

Only imported if config.backend == 'anthropic'. Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=...   (PowerShell: $env:ANTHROPIC_API_KEY="...")

Note: this backend sends text prompts only. To make the doctors reason over the
actual nodule image you would attach the cropped image as an image content
block here (see the Anthropic messages API `image` content type).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .base import LLMBackend

# JSON schema hints per structured task (kept minimal / provider-agnostic).
_SCHEMhints = {
    "judge_nodule": '{"opinion": 0 or 1, "confidence": 0..1}',
    "specialist_opinion": ('{"probs": {"pre_invasive": p, "minimally_invasive": p, '
                           '"invasive": p}, "rationale": str, "cited_features": [str], '
                           '"confidence": 0..1}'),
    "devils_advocate": ('{"probs": {"pre_invasive": p, "minimally_invasive": p, '
                        '"invasive": p}, "challenged_label": str, "alternative": str, '
                        '"rationale": str}'),
}


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-sonnet-5"):
        try:
            import anthropic  # noqa
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic to use the anthropic backend") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        import anthropic
        self._client = anthropic.Anthropic()
        self.model = model

    def _call(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    def complete(self, system: str, prompt: str,
                 context: Optional[Dict[str, Any]] = None) -> str:
        return self._call(system, prompt)

    def structured(self, system: str, prompt: str, task: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        hint = _SCHEMhints.get(task, "{}")
        full = (f"{prompt}\n\nRespond with ONLY valid JSON matching this schema:\n{hint}")
        raw = self._call(system + " You always answer with strict JSON.", full)
        return _extract_json(raw)


def _extract_json(text: str) -> Dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(text[start:end + 1])
