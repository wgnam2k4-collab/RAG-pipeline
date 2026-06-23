"""
embedding/huggingface_embedder.py
==================================
HuggingFace sentence-transformers — dùng cho BAAI/bge-m3.

Cài đặt:
    pip install langchain-huggingface sentence-transformers
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Parameters
    ----------
    model_name           : HuggingFace model identifier. Mặc định BAAI/bge-m3.
    device               : \"cpu\" | \"cuda\" | \"mps\"
    normalize_embeddings : L2-normalise vectors (recommended for cosine similarity).
    query_instruction    : Instruction prefix for queries (instruction-following models).
    document_instruction : Instruction prefix for documents.
    trust_remote_code    : Bật cho các model cần custom code (Qwen3).
    hf_token             : HuggingFace token (cho gated models).
    torch_dtype_str      : \"auto\" | \"float32\" | \"float16\" | \"bfloat16\"
    batch_size           : Số text xử lý song song.
    """

    def __init__(
        self,
        model_name:           str  = "BAAI/bge-m3",
        device:               str  = "cpu",
        normalize_embeddings: bool = True,
        query_instruction:    str | None = None,
        document_instruction: str | None = None,
        trust_remote_code:    bool = False,
        hf_token:             str | None = None,
        torch_dtype_str:      str = "auto",
        batch_size:           int = 32,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.device               = device
        self.normalize_embeddings = normalize_embeddings
        self.query_instruction    = query_instruction
        self.document_instruction = document_instruction
        self.trust_remote_code    = trust_remote_code
        self.hf_token             = hf_token
        self.torch_dtype_str      = torch_dtype_str
        self.batch_size           = batch_size

    def _build(self) -> Embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings

        # Resolve torch dtype
        import os
        _dtype_map = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}
        if self.torch_dtype_str == "auto":
            resolved_dtype = "float16" if self.device == "cuda" else "float32"
        else:
            resolved_dtype = _dtype_map.get(self.torch_dtype_str, "float32")

        dtype_obj = None
        try:
            import torch
            dtype_obj = getattr(torch, resolved_dtype)
        except Exception:
            pass

        model_kw: dict = {"device": self.device}
        if self.trust_remote_code:
            model_kw["trust_remote_code"] = True
        if self.hf_token:
            model_kw["token"] = self.hf_token
        if dtype_obj is not None and self.device == "cuda":
            model_kw["model_kwargs"] = {"torch_dtype": dtype_obj}

        encode_kw: dict = {
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size":           self.batch_size,
        }

        init_kwargs: dict = {
            "model_name":    self.model_name,
            "model_kwargs":  model_kw,
            "encode_kwargs": encode_kw,
        }

        if self.query_instruction:
            init_kwargs["query_instruction"]  = self.query_instruction
        if self.document_instruction:
            init_kwargs["embed_instruction"] = self.document_instruction

        return HuggingFaceEmbeddings(**init_kwargs)
