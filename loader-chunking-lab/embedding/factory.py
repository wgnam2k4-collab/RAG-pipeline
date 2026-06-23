"""
embedding/factory.py
====================
Factory function — chỉ hỗ trợ 2 provider:
  - huggingface : BAAI/bge-m3
  - ollama      : qwen3-embedding:8b
"""

from __future__ import annotations

import logging
from typing import Any

from embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, tuple[str, str]] = {
    "huggingface": ("embedding.huggingface_embedder", "HuggingFaceEmbedder"),
    "ollama":      ("embedding.ollama_embedder",      "OllamaEmbedder"),
}


def get_embedder(provider: str, model_name: str, **kwargs: Any) -> BaseEmbedder:
    """
    Instantiate a dense embedder by provider name.

    Parameters
    ----------
    provider   : \"huggingface\" hoặc \"ollama\".
    model_name : Model identifier.
    **kwargs   : Constructor arguments forwarded to the embedder class.

    Examples
    --------
    >>> get_embedder("huggingface", "BAAI/bge-m3", device="cuda")
    >>> get_embedder("ollama", "qwen3-embedding:8b", base_url="http://localhost:11434")
    """
    entry = _REGISTRY.get(provider)
    if entry is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown embedding provider '{provider}'. Valid: {valid}")

    import importlib
    module_path, class_name = entry
    cls = getattr(importlib.import_module(module_path), class_name)

    logger.info("Embedding: provider=%s  model=%s", provider, model_name)
    return cls(model_name=model_name, **kwargs)
