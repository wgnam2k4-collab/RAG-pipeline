"""
vector_db/faiss_store.py
=========================
FAISS local vector store.

Runs entirely in-process — no server, no API key, no network.
Index được lưu tại:
    <persist_dir>/index.faiss
    <persist_dir>/index.pkl
    <persist_dir>/fingerprint.json

Cài đặt:
    pip install faiss-cpu  (CPU)
    pip install faiss-gpu  (GPU với CUDA)
    pip install langchain-community
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore
from vector_db.utils import corpus_changed, corpus_fingerprint, save_fingerprint

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    Parameters
    ----------
    collection_name : Tên collection (dùng để đặt tên thư mục).
    persist_dir     : Thư mục lưu FAISS index.
    force_reindex   : Xây dựng lại index dù corpus không đổi.
    """

    def __init__(
        self,
        collection_name: str  = "rag_lab",
        persist_dir:     str  = "./storage/faiss",
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.persist_dir = persist_dir

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        lc_embedder = self._langchain_embedder(embedder)
        idx_dir     = Path(self.persist_dir)
        fp_path     = str(idx_dir / "fingerprint.json")

        # Load existing index if corpus is unchanged
        if idx_dir.exists() and not self.force_reindex and not corpus_changed(chunks, fp_path):
            logger.info("FAISS: loading existing index from '%s'.", idx_dir)
            return FAISS.load_local(
                folder_path=str(idx_dir),
                embeddings=lc_embedder,
                allow_dangerous_deserialization=True,
            )

        # Build new index
        logger.info("FAISS: building index for %d chunks.", len(chunks))
        store = FAISS.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
        )

        idx_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(idx_dir))
        save_fingerprint(corpus_fingerprint(chunks), fp_path)
        logger.info("FAISS: index saved to '%s'.", idx_dir)
        return store
