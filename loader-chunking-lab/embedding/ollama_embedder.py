"""
embedding/ollama_embedder.py
=============================
Embedding via a locally running Ollama server.
Dùng cho qwen3-embedding:8b.

Pull model trước:
    ollama pull qwen3-embedding:8b
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """
    Parameters
    ----------
    model_name : Ollama model name (đã pull sẵn). Mặc định qwen3-embedding:8b.
    base_url   : Ollama server URL (default: http://localhost:11434).
    """

    def __init__(
        self,
        model_name: str = "qwen3-embedding:8b",
        base_url:   str = "http://localhost:11434",
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url

    def _build(self) -> Embeddings:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=self.model_name,
            base_url=self.base_url,
        )
