"""LLM backend factory."""
from __future__ import annotations

from ..config import Config
from .base import LLMBackend
from .mock import MockBackend


def get_backend(config: Config) -> LLMBackend:
    if config.backend == "mock":
        return MockBackend(seed_base=config.seed)
    if config.backend == "anthropic":
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(model=config.model)
    if config.backend == "openai":
        from .openai_backend import OpenAIBackend
        return OpenAIBackend(model=config.model)
    raise ValueError(f"Unknown backend: {config.backend!r}")
