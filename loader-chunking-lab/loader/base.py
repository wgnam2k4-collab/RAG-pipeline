"""
loader/base.py
==============
Abstract base class cho tất cả document loader.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from langchain_core.documents import Document

Language = Literal["vi", "en", "both"]


class BaseLoader(ABC):
    """
    Base class cho mọi document loader.

    Tham số
    -------
    language : Ngôn ngữ tài liệu nguồn.
               "vi" = Tiếng Việt, "en" = Tiếng Anh, "both" = hỗn hợp.
    """

    def __init__(self, language: Language = "both"):
        self.language = language

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """Load một file và trả về danh sách Document."""

    def _stamp(self, docs: list[Document]) -> list[Document]:
        """Gắn ngôn ngữ vào metadata của mỗi document (in-place)."""
        for doc in docs:
            doc.metadata.setdefault("language", self.language)
        return docs
