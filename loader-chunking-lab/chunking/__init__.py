"""
chunking/__init__.py
"""
from chunking.factory       import get_chunker, chunk_documents_from_config
from chunking.deduplication import deduplicate_chunks

__all__ = ["get_chunker", "chunk_documents_from_config", "deduplicate_chunks"]
