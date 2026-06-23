"""
pipeline_cache.py
=================
Cache cấp bước (step-level) cho RAG pipeline, lưu trên disk.

Cơ chế: Pipeline Fingerprint Chain
───────────────────────────────────
Mỗi bước có step_key = SHA256(prev_step_key + canonical_json(step_config)).

  input_hash ──► loader_key ──► chunking_key ──► embedding_key ──► vdb_key

Tính chất:
- Input + loader config không đổi              → loader cache HIT
- Input không đổi, chunking config thay đổi   → chunking MISS, loader HIT
- Input thay đổi                               → tất cả MISS

Cấu trúc thư mục:
  processed_data/
    <input_hash_12chars>/
      input_meta.json
      loader/<loader_key_12chars>/{docs.pkl, meta.json}
      chunking/<chunk_key_12chars>/{chunks.pkl, meta.json}
      embedding/<embed_key_12chars>/{dense.npz, sparse.pkl?, meta.json}
      vector_db/<vdb_key_12chars>/meta.json   # chỉ metadata, data ở persist_dir

Dùng tên 12 ký tự đầu của SHA256 (~69 nghìn tỷ khả năng) để đủ unique và dễ đọc.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
KEY_LEN       = 12        # hex chars to use from SHA256 (12 = 48 bits, ~281T combos)
CHUNK_SIZE    = 65_536    # bytes per read when hashing large files
STEP_NAMES    = ("loader", "chunking", "embedding", "vector_db")   # ordered — used for invalidation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    """SHA256 of a single file, streamed (memory-efficient for large files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON — sorted keys, no whitespace, None→null."""
    def _clean(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in sorted(o.items())}
        if isinstance(o, (list, tuple)):
            return [_clean(i) for i in o]
        if isinstance(o, Path):
            return str(o)
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        return str(o)   # fallback — e.g. Enum, custom types
    return json.dumps(_clean(obj), ensure_ascii=False, separators=(",", ":"))


def _short(full_hash: str) -> str:
    return full_hash[:KEY_LEN]


# ── Main class ────────────────────────────────────────────────────────────────

class PipelineCache:
    """
    Disk-backed cache for RAG pipeline steps.

    Thread-safe for reads; writes use atomic rename to avoid partial files.
    """

    def __init__(self, base_dir: str | Path = "processed_data"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        _write_gitignore(self.base)

    # ── Hash / key computation ─────────────────────────────────────────────

    def compute_input_hash(self, source_path: str) -> str:
        """
        Content-based SHA256 hash của toàn bộ file(s) từ source_path.

        Chỉ dùng tên file (p.name) + nội dung — KHÔNG dùng đường dẫn thư mục.
        Lý do: file upload vào Streamlit được lưu vào thư mục temp ngẫu nhiên
        (tmpXXXXXX) mỗi lần → nếu hash cả path thì cùng một file upload lại
        sẽ cho hash khác → cache miss sai.

        Tính chất:
          - Cùng file, upload lại           → hash GIỐNG (cache HIT ✅)
          - Cùng tên nhưng nội dung khác    → hash KHÁC  (cache MISS ✅)
          - Khác tên, cùng nội dung         → hash KHÁC  (cache MISS ✅)
          - Nhiều file: sort theo tên để deterministic

        source_path có thể là:
          - Đường dẫn file đơn
          - Đường dẫn thư mục (đệ quy)
          - Nhiều file cách nhau dấu phẩy  "a.pdf,b.docx"
        """
        h = hashlib.sha256()
        paths = _resolve_paths(source_path)
        # Sort theo tên file (không phải full path) để deterministic
        for p in sorted(paths, key=lambda x: x.name):
            if p.is_file():
                h.update(p.name.encode())           # tên file
                h.update(_sha256_file(p).encode())  # nội dung
        return h.hexdigest()

    def make_step_key(self, prev_key: str, step_cfg: dict) -> str:
        """
        step_key = SHA256(prev_key + canonical_json(step_cfg))

        Tính chất:
          - prev_key thay đổi → step_key thay đổi (propagation)
          - step_cfg thay đổi → step_key thay đổi
          - Cùng prev_key + cùng cfg → cùng step_key (deterministic)
        """
        combined = prev_key + _canonical_json(step_cfg)
        return _sha256_str(combined)

    # ── Directory helpers ─────────────────────────────────────────────────

    def _input_dir(self, input_hash: str) -> Path:
        return self.base / _short(input_hash)

    def _step_dir(self, input_hash: str, step: str, step_key: str) -> Path:
        return self._input_dir(input_hash) / step / _short(step_key)

    def has_step(self, input_hash: str, step: str, step_key: str) -> bool:
        """True nếu bước đã được cache."""
        d = self._step_dir(input_hash, step, step_key)
        return (d / "meta.json").exists()

    # ── Loader ────────────────────────────────────────────────────────────

    def save_loader(
        self,
        input_hash:    str,
        loader_key:    str,
        docs:          list,
        cfg:           dict,
        source_path:   str = "",
        input_display: str = "",
    ) -> None:
        d = self._step_dir(input_hash, "loader", loader_key)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_pickle(docs, d / "docs.pkl")
        _write_meta(d, step="loader", cfg=cfg, stats={
            "n_docs": len(docs),
            "n_chars": sum(len(doc.page_content) for doc in docs),
            "source_path": source_path,
        }, parent_key=input_hash)
        # Lưu input meta — cập nhật input_display nếu có
        _write_input_meta(self._input_dir(input_hash), input_hash, source_path, input_display)
        _stamp_size(d)
        logger.debug("Cache SAVE loader %s", _short(loader_key))

    def load_loader(self, input_hash: str, loader_key: str) -> list | None:
        d = self._step_dir(input_hash, "loader", loader_key)
        pkl = d / "docs.pkl"
        if not pkl.exists():
            return None
        try:
            with open(pkl, "rb") as f:
                docs = pickle.load(f)
            logger.debug("Cache HIT  loader %s", _short(loader_key))
            return docs
        except Exception as e:
            logger.warning("Cache corrupt loader %s: %s — removing", _short(loader_key), e)
            shutil.rmtree(d, ignore_errors=True)
            return None

    # ── Chunking ──────────────────────────────────────────────────────────

    def save_chunking(
        self,
        input_hash: str,
        chunk_key: str,
        chunks: list,
        cfg: dict,
        loader_key: str = "",
    ) -> None:
        d = self._step_dir(input_hash, "chunking", chunk_key)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_pickle(chunks, d / "chunks.pkl")
        _write_meta(d, step="chunking", cfg=cfg, stats={
            "n_chunks":    len(chunks),
            "avg_chars":   int(sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)),
            "total_chars": sum(len(c.page_content) for c in chunks),
        }, parent_key=loader_key)
        _stamp_size(d)
        logger.debug("Cache SAVE chunking %s", _short(chunk_key))

    def load_chunking(self, input_hash: str, chunk_key: str) -> list | None:
        d = self._step_dir(input_hash, "chunking", chunk_key)
        pkl = d / "chunks.pkl"
        if not pkl.exists():
            return None
        try:
            with open(pkl, "rb") as f:
                chunks = pickle.load(f)
            logger.debug("Cache HIT  chunking %s", _short(chunk_key))
            return chunks
        except Exception as e:
            logger.warning("Cache corrupt chunking %s: %s — removing", _short(chunk_key), e)
            shutil.rmtree(d, ignore_errors=True)
            return None

    # ── Embedding ─────────────────────────────────────────────────────────

    def save_embedding(
        self,
        input_hash: str,
        embed_key: str,
        result: dict,
        cfg: dict,
        chunk_key: str = "",
    ) -> None:
        """
        result = {"dense": list[list[float]], "sparse": ..., "dims": int, ...}
        dense lưu dưới dạng .npz (float32) — tiết kiệm hơn pickle ~4×.
        """
        import numpy as np

        d = self._step_dir(input_hash, "embedding", embed_key)
        d.mkdir(parents=True, exist_ok=True)

        # Dense → .npz
        dense_arr = np.array(result["dense"], dtype=np.float32)
        np.savez_compressed(d / "dense.npz", dense=dense_arr)

        # Sparse → pickle (dict list, không dễ serialize sang npz)
        if result.get("sparse") is not None:
            _atomic_pickle(result["sparse"], d / "sparse.pkl")

        _write_meta(d, step="embedding", cfg=cfg, stats={
            "n_vectors":   result["n_embedded"],
            "dims":        result["dims"],
            "has_sparse":  result.get("sparse") is not None,
            "truncated":   result.get("truncated", False),
        }, parent_key=chunk_key)
        _stamp_size(d)
        logger.debug("Cache SAVE embedding %s", _short(embed_key))

    def load_embedding(self, input_hash: str, embed_key: str) -> dict | None:
        import numpy as np

        d = self._step_dir(input_hash, "embedding", embed_key)
        npz = d / "dense.npz"
        if not npz.exists():
            return None
        try:
            data      = np.load(npz)
            dense     = data["dense"].tolist()
            sparse    = None
            sparse_f  = d / "sparse.pkl"
            if sparse_f.exists():
                with open(sparse_f, "rb") as f:
                    sparse = pickle.load(f)
            meta      = _read_meta(d)
            stats     = meta.get("stats", {})
            logger.debug("Cache HIT  embedding %s", _short(embed_key))
            return {
                "dense":      dense,
                "sparse":     sparse,
                "dims":       stats.get("dims", len(dense[0]) if dense else 0),
                "n_embedded": stats.get("n_vectors", len(dense)),
                "truncated":  stats.get("truncated", False),
            }
        except Exception as e:
            logger.warning("Cache corrupt embedding %s: %s — removing", _short(embed_key), e)
            shutil.rmtree(d, ignore_errors=True)
            return None

    # ── Vector DB ─────────────────────────────────────────────────────────

    def save_vector_db(
        self,
        input_hash: str,
        vdb_key:    str,
        result:     dict,
        cfg:        dict,
        embed_key:  str = "",
    ) -> None:
        d = self._step_dir(input_hash, "vector_db", vdb_key)
        d.mkdir(parents=True, exist_ok=True)
        _write_meta(d, step="vector_db", cfg=cfg, stats={
            "provider":        result["provider"],
            "collection_name": result["collection_name"],
            "n_vectors":       result["n_vectors"],
            "persist_dir":     result.get("persist_dir", cfg.get("persist_dir", "")),
        }, parent_key=embed_key)
        _stamp_size(d)
        logger.debug("Cache SAVE vector_db %s", _short(vdb_key))

    def load_vector_db(self, input_hash: str, vdb_key: str) -> dict | None:
        d = self._step_dir(input_hash, "vector_db", vdb_key)
        if not (d / "meta.json").exists():
            return None
        try:
            meta = _read_meta(d)
            logger.debug("Cache HIT  vector_db %s", _short(vdb_key))
            return {
                **meta.get("stats", {}),
                "cfg":      meta.get("cfg", {}),
                "saved_at": meta.get("saved_at", ""),
            }
        except Exception as e:
            logger.warning("Cache corrupt vector_db %s: %s — removing", _short(vdb_key), e)
            shutil.rmtree(d, ignore_errors=True)
            return None

    # ── Cache info / management ───────────────────────────────────────────

    def list_entries(self) -> list[dict]:
        entries = []
        if not self.base.exists():
            return entries
        for input_dir in sorted(self.base.iterdir()):
            if not input_dir.is_dir() or input_dir.name.startswith("."):
                continue
            meta_f = input_dir / "input_meta.json"
            meta = json.loads(meta_f.read_text("utf-8")) if meta_f.exists() else {}
            steps: dict[str, list[dict]] = {}
            total_size_mb = 0.0
            for step in STEP_NAMES:
                step_dir = input_dir / step
                if not step_dir.exists():
                    continue
                steps[step] = []
                for key_dir in sorted(step_dir.iterdir()):
                    if not key_dir.is_dir():
                        continue
                    m        = _read_meta(key_dir)
                    size_mb  = m.get("size_mb", 0.0)
                    total_size_mb += size_mb
                    steps[step].append({
                        "key_short": key_dir.name,
                        "cfg":       m.get("cfg", {}),
                        "stats":     m.get("stats", {}),
                        "saved_at":  m.get("saved_at", ""),
                        "size_mb":   size_mb,
                    })
            entries.append({
                "input_short":   input_dir.name,
                "full_hash":     meta.get("full_hash", ""),
                "source_path":   meta.get("source_path", ""),
                "input_display": meta.get("input_display", "") or meta.get("source_path", ""),
                "created_at":    meta.get("created_at", ""),
                "steps":         steps,
                "total_size_mb": round(total_size_mb, 4),
            })
        return entries

    def total_size_mb(self) -> float:
        return _dir_size_mb(self.base)

    def clear_all(self) -> None:
        """Xoá toàn bộ cache."""
        if self.base.exists():
            shutil.rmtree(self.base)
        self.base.mkdir(parents=True, exist_ok=True)
        _write_gitignore(self.base)

    def clear_input(self, input_short: str) -> None:
        """Xoá cache của một input cụ thể."""
        d = self.base / input_short
        if d.exists():
            shutil.rmtree(d)


# ── Private helpers ───────────────────────────────────────────────────────────

def _resolve_paths(source_path: str) -> list[Path]:
    """Trả về list[Path] từ source_path (file, dir, hoặc comma-separated)."""
    paths: list[Path] = []
    for part in source_path.split(","):
        p = Path(part.strip())
        if p.is_dir():
            paths.extend(p.rglob("*"))
        elif p.is_file():
            paths.append(p)
    return [p for p in paths if p.is_file()]


def _atomic_pickle(obj: Any, dest: Path) -> None:
    """Write pickle atomically: write to .tmp → rename."""
    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(dest)


def _write_meta(d: Path, step: str, cfg: dict, stats: dict, parent_key: str = "") -> None:
    meta = {
        "step":       step,
        "cfg":        cfg,
        "stats":      stats,
        "saved_at":   datetime.now().isoformat(timespec="seconds"),
        "parent_key": parent_key,
    }
    data = json.dumps(meta, ensure_ascii=False, indent=2)
    dest = d / "meta.json"
    tmp  = dest.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(dest)


def _stamp_size(d: Path) -> None:
    """Cập nhật size_mb vào meta.json sau khi tất cả files đã được ghi."""
    meta_f = d / "meta.json"
    if not meta_f.exists():
        return
    try:
        meta = json.loads(meta_f.read_text("utf-8"))
        meta["size_mb"] = round(
            sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024**2), 4
        )
        tmp = meta_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(meta_f)
    except Exception:
        pass


def _read_meta(d: Path) -> dict:
    f = d / "meta.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text("utf-8"))
    except Exception:
        return {}


def _write_input_meta(
    input_dir:     Path,
    full_hash:     str,
    source_path:   str,
    input_display: str = "",
) -> None:
    f = input_dir / "input_meta.json"
    input_dir.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if f.exists():
        try:
            existing = json.loads(f.read_text("utf-8"))
        except Exception:
            existing = {}
    meta = {
        "full_hash":     full_hash,
        "short_hash":    full_hash[:KEY_LEN],
        "source_path":   source_path,
        "input_display": input_display or existing.get("input_display", "") or source_path,
        "created_at":    existing.get("created_at", datetime.now().isoformat(timespec="seconds")),
    }
    tmp = input_dir / "input_meta.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(f)


def _dir_size_mb(d: Path) -> float:
    if not d.exists():
        return 0.0
    total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return round(total / (1024 ** 2), 2)


def _write_gitignore(base: Path) -> None:
    gi = base / ".gitignore"
    if not gi.exists():
        gi.write_text(
            "# Auto-generated by pipeline_cache.py\n"
            "# Ignore large processed data files\n"
            "*.pkl\n"
            "*.npz\n",
            encoding="utf-8",
        )
