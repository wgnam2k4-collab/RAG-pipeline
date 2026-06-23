"""
app.py — Loader & Chunking Lab
===============================
Giao diện Streamlit độc lập để thử nghiệm tất cả các kỹ thuật
Loader và Chunking từ RAG-pipeline-visualizer.

Khởi động:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import re
import json
import tempfile
import hashlib
import time
from pathlib import Path
from pipeline_cache import PipelineCache

# CUDA allocator phải set trước khi bất kỳ torch import nào
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Đảm bảo project root trong sys.path
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv as _ld
    _ld(override=True)
except ImportError:
    pass

import streamlit as st

# ─── Constants ────────────────────────────────────────────────────────────────
APP_TITLE = "🔬 Loader & Chunking Lab"

PDF_STRATEGIES = {
    "pypdf":          "Nhanh, text layer, không cần dep thêm",
    "pymupdf":        "Nhanh hơn, layout tốt hơn · pip install pymupdf",
    "pdfplumber":     "Trích bảng tốt nhất (text layer) · pip install pdfplumber",
    "unstructured":   "OCR + bảng + hình (tốt nhất) · pip install 'unstructured[pdf]'",
    "docling":        "IBM parser, Markdown output xuất sắc · pip install docling",
    "marker":         "Markdown chất lượng cao, bảng & LaTeX · pip install marker-pdf",
    "opendataloader": "#1 benchmark (0.90), bounding box · pip install opendataloader-pdf (Java 11+)",
}

CHUNKING_STRATEGIES = {
    "recursive":      "Cắt theo đoạn → dòng → câu → ký tự. Mặc định tốt nhất.",
    "token_based":    "Đếm token (BPE). Quan trọng với tiếng Việt & embedding limit.",
    "format_aware":   "Cắt theo cấu trúc tài liệu (Markdown heading, code block, HTML).",
    "sentence_aware": "Cắt theo ranh giới câu. Tốt cho Q&A, FAQ.",
    "semantic":       "Cắt theo cosine similarity. Tốt cho PDF nhiều chủ đề.",
    "hierarchical":   "Parent (section lớn) + child (đoạn nhỏ). Giảm hallucination.",
    "contextual":     "Recursive + LLM thêm context prefix mỗi chunk (Anthropic method).",
}

_LOADER_PAGE_SIZE   = 15
_CHUNKING_PAGE_SIZE = 25


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_gpu() -> tuple[str, str]:
    """Trả về (device, label) của GPU tốt nhất có sẵn."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return "cuda", f"CUDA · {name}"
        if torch.backends.mps.is_available():
            return "mps", "MPS (Apple Silicon)"
    except ImportError:
        pass
    return "cpu", "CPU only"


def _is_installed(pkg: str) -> bool:
    """Kiểm tra package đã cài chưa."""
    import importlib.util
    return importlib.util.find_spec(pkg) is not None


def _check_strategy_installed(strategy: str) -> tuple[bool, str]:
    """Trả về (installed, hint) cho từng PDF strategy."""
    checks = {
        "pypdf":          ("pypdf",          "pip install pypdf"),
        "pymupdf":        ("fitz",           "pip install pymupdf"),
        "pdfplumber":     ("pdfplumber",     "pip install pdfplumber"),
        "unstructured":   ("unstructured",   "pip install 'unstructured[pdf]'"),
        "docling":        ("docling",        "pip install docling"),
        "marker":         ("marker",         "pip install marker-pdf"),
        "opendataloader": ("opendataloader_pdf", "pip install opendataloader-pdf"),
    }
    pkg, hint = checks.get(strategy, ("", ""))
    if not pkg:
        return True, ""
    return _is_installed(pkg), hint


def _file_type_badge(ft: str) -> str:
    colors = {
        "pdf": ("#ff4b4b", "#fff"),
        "markdown": ("#0066cc", "#fff"),
        "html": ("#e34c26", "#fff"),
        "code": ("#4CAF50", "#fff"),
        "text": ("#888", "#fff"),
    }
    bg, fg = colors.get(ft, ("#888", "#fff"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:12px;font-size:13px;font-weight:600;">{ft.upper()}</span>'
    )


def _chunk_level_badge(level: str) -> str:
    if level == "parent":
        return '<span style="background:#7b2ff7;color:#fff;padding:2px 9px;border-radius:10px;font-size:12px;">PARENT</span>'
    if level == "child":
        return '<span style="background:#00b4d8;color:#fff;padding:2px 10px;border-radius:10px;font-size:12px;">CHILD</span>'
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Image rendering helpers (kế thừa từ RAG-pipeline-visualizer)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_images_in_text(content: str, area_key: str, area_height: int):
    IMAGE_RE = re.compile(
        r'!\[([^\]]*)\]\((data:image/[^)]+|[^)]+\.(jpe?g|png|webp|gif|bmp))\)',
        re.IGNORECASE,
    )
    if not IMAGE_RE.search(content):
        st.text_area(
            "Nội dung", value=content, height=area_height,
            key=area_key, disabled=True, label_visibility="collapsed",
        )
        return

    parts   = IMAGE_RE.split(content)
    matches = IMAGE_RE.findall(content)

    text_before = parts[0]
    if text_before.strip():
        st.text_area(
            "Nội dung", value=text_before,
            height=min(area_height, max(60, len(text_before) // 3)),
            key=f"{area_key}_t0", disabled=True, label_visibility="collapsed",
        )

    for m_idx, (alt, img_path, _ext) in enumerate(matches):
        caption = alt or "Figure"
        if img_path.startswith("data:"):
            import base64 as _b64
            try:
                header, b64data = img_path.split(",", 1)
                img_bytes = _b64.b64decode(b64data)
                st.image(img_bytes, caption=caption, use_container_width=True)
            except Exception:
                st.caption("🖼️ *(không render được data URI)*")
        else:
            img_file = Path(img_path)
            if img_file.exists():
                st.image(str(img_file), caption=caption, use_container_width=True)
            else:
                st.caption(f"🖼️ `{img_path}` *(ảnh không tìm thấy trên disk)*")

        text_after_idx = 1 + m_idx * 4 + 3
        if text_after_idx < len(parts):
            text_after = parts[text_after_idx]
            if text_after.strip():
                st.text_area(
                    "Nội dung", value=text_after,
                    height=min(area_height, max(60, len(text_after) // 3)),
                    key=f"{area_key}_t{m_idx+1}", disabled=True, label_visibility="collapsed",
                )


def _local_images_to_base64(content: str) -> str:
    import base64, mimetypes
    IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    def _to_data_uri(m: re.Match) -> str:
        alt = m.group(1)
        src = m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        img_path = Path(src)
        if not img_path.exists():
            return m.group(0)
        mime, _ = mimetypes.guess_type(str(img_path))
        mime = mime or "image/jpeg"
        b64  = base64.b64encode(img_path.read_bytes()).decode()
        return f"![{alt}](data:{mime};base64,{b64})"

    return IMAGE_RE.sub(_to_data_uri, content)


def render_content_with_images(content: str):
    if r"\n" in content and "\n" not in content:
        content = content.replace(r"\n", "\n")
    md_with_embedded = _local_images_to_base64(content)
    st.markdown(md_with_embedded, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Loader Settings Panel
# ═══════════════════════════════════════════════════════════════════════════════

def render_loader_settings() -> dict:
    """Render sidebar settings cho loader, trả về dict params."""
    st.subheader("⚙️ Cài đặt Loader")

    col1, col2 = st.columns(2)
    with col1:
        pdf_strategy = st.selectbox(
            "PDF Strategy",
            options=list(PDF_STRATEGIES.keys()),
            key="sel_pdf_strategy",
            help="\n\n".join(f"**{k}** — {v}" for k, v in PDF_STRATEGIES.items()),
        )
        installed, hint = _check_strategy_installed(pdf_strategy)
        if not installed:
            st.caption(f"⚠️ Cần cài: `{hint}`")

        extract_tables = st.checkbox("Trích xuất bảng → Markdown", value=True)

    with col2:
        language = st.selectbox(
            "Ngôn ngữ corpus",
            options=["both", "vi", "en"],
            index=0,
            help="Dùng để chọn ngôn ngữ cho OCR và NLP tools",
        )

    # Strategy info
    if PDF_STRATEGIES.get(pdf_strategy):
        st.info(f"ℹ️ **{pdf_strategy}** — {PDF_STRATEGIES[pdf_strategy]}")

    # ── Marker device + VLM ─────────────────────────────────────────────────
    marker_device   = "cpu"
    describe_images = False
    vision_provider = "openai"
    vision_model    = "gpt-4o-mini"
    ollama_base_url = "http://localhost:11434/v1"
    odl_hybrid      = None
    odl_struct_tree = False

    if pdf_strategy == "marker":
        st.markdown("---")
        st.subheader("🖥️ Thiết bị cho Marker")
        auto_device, gpu_label = _detect_gpu()
        if auto_device != "cpu":
            st.success(f"✅ Phát hiện GPU: {gpu_label}")
        else:
            st.warning("⚠️ Không phát hiện GPU — Marker sẽ chạy trên CPU (~30-60s/trang)")

        device_options = [auto_device, "cpu"] if auto_device != "cpu" else ["cpu"]
        marker_device = st.radio(
            "Chọn device", options=device_options, index=0, horizontal=True,
            help="cuda: NVIDIA GPU · mps: Apple Silicon · cpu: chậm nhất",
        )
        st.markdown("---")
        st.markdown("**🖼️ Mô tả ảnh bằng VLM**")
        describe_images = st.checkbox("Dùng VLM mô tả ảnh trong PDF", value=False, key="_vlm_describe_marker")
        if describe_images:
            vision_provider = st.radio("VLM Provider", ["openai", "ollama"], horizontal=True, key="_vlm_provider_marker")
            if vision_provider == "openai":
                vision_model = st.selectbox("VLM Model", ["gpt-4o-mini", "gpt-4o"], key="_vlm_model_marker")
            else:
                vision_model = st.selectbox(
                    "VLM Model (Ollama)",
                    ["llava:7b", "llava:13b", "llava-llama3", "minicpm-v", "qwen2-vl:7b"],
                    key="_vlm_model_marker",
                )
                ollama_base_url = st.text_input("Ollama base URL", value="http://localhost:11434/v1", key="_vlm_url_marker")

    elif pdf_strategy == "unstructured":
        st.markdown("---")
        st.markdown("**🖼️ Mô tả ảnh bằng VLM**")
        describe_images = st.checkbox("Dùng VLM mô tả ảnh trong PDF", value=False, key="_vlm_describe_unst")
        if describe_images:
            vision_provider = st.radio("VLM Provider", ["openai", "ollama"], horizontal=True, key="_vlm_provider_unst")
            if vision_provider == "openai":
                vision_model = st.selectbox("VLM Model", ["gpt-4o-mini", "gpt-4o"], key="_vlm_model_unst")
            else:
                vision_model = st.selectbox(
                    "VLM Model (Ollama)",
                    ["llava:7b", "llava:13b", "minicpm-v", "qwen2-vl:7b"],
                    key="_vlm_model_unst",
                )
                ollama_base_url = st.text_input("Ollama base URL", value="http://localhost:11434/v1", key="_vlm_url_unst")

    elif pdf_strategy == "opendataloader":
        st.markdown("---")
        st.subheader("⚙️ Cài đặt OpenDataLoader")
        odl_mode = st.radio(
            "Mode", options=["fast", "hybrid"], index=0, horizontal=True, key="sel_odl_mode",
            help="**fast** — Java local, 0.05s/trang, accuracy 0.72.\n\n**hybrid** — AI routing, accuracy 0.90 (#1 benchmark).",
        )
        if odl_mode == "hybrid":
            odl_hybrid = "docling-fast"
            st.caption("Cần khởi động server: `opendataloader-pdf-hybrid --port 5002`")
        odl_struct_tree = st.checkbox("Dùng native PDF structure tags", value=False, key="odl_struct_tree")

    elif pdf_strategy == "docling":
        st.markdown("---")
        st.info(
            "🖼️ **Docling tự động xử lý ảnh** — Docling extract ảnh và nhúng vào Markdown output. "
            "Không cần VLM."
        )

    return {
        "pdf_strategy":    pdf_strategy,
        "extract_tables":  extract_tables,
        "language":        language,
        "marker_device":   marker_device,
        "describe_images": describe_images,
        "vision_model":    vision_model,
        "vision_provider": vision_provider,
        "ollama_base_url": ollama_base_url,
        "odl_hybrid":      odl_hybrid,
        "odl_struct_tree": odl_struct_tree,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Chunking Settings Panel
# ═══════════════════════════════════════════════════════════════════════════════

def render_chunking_settings() -> tuple[str, int, int, dict]:
    """Render sidebar settings cho chunking, trả về (strategy, size, overlap, extra)."""
    st.subheader("✂️ Cài đặt Chunking")

    strategy = st.selectbox(
        "Chunking Strategy",
        options=list(CHUNKING_STRATEGIES.keys()),
        key="sel_chunking_strategy",
        format_func=lambda s: f"{s}  —  {CHUNKING_STRATEGIES[s][:50]}…" if len(CHUNKING_STRATEGIES[s]) > 50 else f"{s}  —  {CHUNKING_STRATEGIES[s]}",
    )

    if CHUNKING_STRATEGIES.get(strategy):
        st.info(f"ℹ️ {CHUNKING_STRATEGIES[strategy]}")

    _NO_SIZE = strategy == "semantic"
    _DEFER   = strategy == "format_aware"

    if not _DEFER:
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk size (chars)", min_value=50, max_value=8000, value=500, step=50,
                disabled=_NO_SIZE,
            )
        with col2:
            chunk_overlap = st.number_input(
                "Chunk overlap (chars)", min_value=0, max_value=2000, value=100, step=25,
                disabled=_NO_SIZE,
            )
        if _NO_SIZE:
            st.caption("ℹ️ Chunk size không áp dụng cho **semantic** — ranh giới do cosine similarity quyết định.")
    else:
        chunk_size    = 1000
        chunk_overlap = 100

    extra: dict = {}

    if strategy == "token_based":
        extra["encoding_name"] = st.selectbox(
            "Tokenizer encoding",
            ["cl100k_base", "p50k_base", "r50k_base"],
            help="cl100k_base: GPT-4 / text-embedding-3",
        )

    elif strategy == "semantic":
        from chunking.semantic import EMBEDDING_MODELS, PROVIDER_GROUPS

        provider_group = st.selectbox(
            "Embedding Provider Group",
            options=list(PROVIDER_GROUPS.keys()),
            index=list(PROVIDER_GROUPS.keys()).index(
                next((g for g in PROVIDER_GROUPS if g.startswith("🏠 Self-hosted · HuggingFace")),
                     list(PROVIDER_GROUPS.keys())[0])
            ),
        )
        models_in_group = PROVIDER_GROUPS[provider_group]

        def _model_label(m: str) -> str:
            meta = EMBEDDING_MODELS.get(m, {})
            return f"{meta.get('display', m)}  ·  MTEB {meta.get('mteb','?')}  ·  {meta.get('dim','?')}d"

        embedding_model = st.selectbox("Embedding Model", options=models_in_group, format_func=_model_label)
        extra["embedding_model_name"] = embedding_model

        meta = EMBEDDING_MODELS.get(embedding_model, {})
        if meta.get("note"):
            st.info(f"ℹ️ {meta['note']}")

        provider = meta.get("provider", "")
        if provider == "ollama":
            ollama_url = st.text_input("Ollama base URL", value="http://localhost:11434/v1", key="ollama_embed_url")
            extra["ollama_base_url"] = ollama_url

        extra["breakpoint_type"] = st.selectbox(
            "Breakpoint detection type",
            ["percentile", "standard_deviation", "interquartile"],
        )
        extra["breakpoint_threshold"] = st.slider(
            "Breakpoint threshold", min_value=50.0, max_value=99.0, value=95.0, step=1.0,
        )

    elif strategy == "format_aware":
        extra["format_type"] = st.selectbox(
            "Format type", ["auto", "markdown", "code", "html"],
            key="sel_format_type",
            help="auto: tự phát hiện từ metadata",
        )
        extra["split_large_sections"] = st.checkbox("Chia nhỏ section quá dài", value=False)
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk size (chars)", min_value=50, max_value=8000, value=1000, step=50,
                disabled=not extra["split_large_sections"], key="fa_chunk_size",
            )
        with col2:
            chunk_overlap = st.number_input(
                "Chunk overlap (chars)", min_value=0, max_value=2000, value=100, step=25,
                disabled=not extra["split_large_sections"], key="fa_chunk_overlap",
            )

    elif strategy == "hierarchical":
        extra["parent_chunk_size"] = st.number_input(
            "Parent chunk size (chars)", min_value=200, max_value=10000, value=2000, step=100,
        )

    elif strategy == "contextual":
        st.warning(
            "⚠️ Contextual chunking cần LLM API (chậm và tốn phí). "
            "Mỗi chunk sẽ được gửi lên LLM để sinh context prefix."
        )
        llm_provider = st.radio("LLM Provider", ["openai", "anthropic", "google"], horizontal=True)
        extra["llm_provider"] = llm_provider

        if llm_provider == "anthropic":
            extra["llm_model"] = st.selectbox(
                "LLM Model", ["claude-haiku-4-5-20251001", "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"]
            )
        elif llm_provider == "openai":
            extra["llm_model"] = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        else:
            extra["llm_model"] = st.selectbox("LLM Model", ["gemini-1.5-flash", "gemini-2.0-flash"])

        extra["base_strategy"] = st.selectbox("Base strategy", ["recursive", "sentence_aware", "token_based"])
        extra["n_sentences"]   = st.number_input("Context sentences per chunk", min_value=1, max_value=5, value=2)

    return strategy, int(chunk_size), int(chunk_overlap), extra


# ═══════════════════════════════════════════════════════════════════════════════
# Run Loader
# ═══════════════════════════════════════════════════════════════════════════════

def run_loader(file_path: str, loader_params: dict) -> list:
    """Chạy loader với params đã chọn."""
    from loader.directory_loader import PDFDocumentLoader

    return PDFDocumentLoader(
        language        = loader_params["language"],
        pdf_strategy    = loader_params["pdf_strategy"],
        extract_tables  = loader_params["extract_tables"],
        extract_images  = loader_params.get("describe_images", False),
        vision_model    = loader_params.get("vision_model", "gpt-4o-mini"),
        deduplicate     = True,
        marker_device   = loader_params.get("marker_device", "cpu"),
        describe_images = loader_params.get("describe_images", False),
        vision_provider = loader_params.get("vision_provider", "openai"),
        ollama_base_url = loader_params.get("ollama_base_url", "http://localhost:11434/v1"),
        odl_hybrid      = loader_params.get("odl_hybrid"),
        odl_struct_tree = loader_params.get("odl_struct_tree", False),
    ).load(file_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Run Chunker
# ═══════════════════════════════════════════════════════════════════════════════

def run_chunker(docs: list, strategy: str, chunk_size: int, chunk_overlap: int, extra: dict) -> list:
    """Chạy chunker với params đã chọn."""
    from chunking.factory import get_chunker

    kwargs = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, **extra}

    # Loại bỏ keys không hợp lệ cho từng strategy
    if strategy == "token_based":
        kwargs = {k: v for k, v in kwargs.items() if k in ("chunk_size", "chunk_overlap", "encoding_name")}
    elif strategy == "semantic":
        kwargs = {k: v for k, v in kwargs.items() if k in (
            "chunk_size", "chunk_overlap", "embedding_model_name",
            "breakpoint_type", "breakpoint_threshold", "ollama_base_url",
        )}
    elif strategy == "format_aware":
        kwargs = {k: v for k, v in kwargs.items() if k in (
            "chunk_size", "chunk_overlap", "format_type", "split_large_sections",
        )}
    elif strategy == "hierarchical":
        kwargs = {k: v for k, v in kwargs.items() if k in (
            "chunk_size", "chunk_overlap", "parent_chunk_size",
        )}
    elif strategy == "contextual":
        base_kwargs = {k: v for k, v in kwargs.items() if k in ("chunk_size", "chunk_overlap")}
        kwargs = {
            "chunk_size":    chunk_size,
            "chunk_overlap": chunk_overlap,
            "base_strategy": extra.get("base_strategy", "recursive"),
            "base_kwargs":   base_kwargs,
            "llm_model":     extra.get("llm_model", "claude-haiku-4-5-20251001"),
            "llm_provider":  extra.get("llm_provider", "anthropic"),
            "n_sentences":   extra.get("n_sentences", 2),
        }
    elif strategy == "sentence_aware":
        kwargs = {k: v for k, v in kwargs.items() if k in ("chunk_size", "chunk_overlap")}
    elif strategy == "recursive":
        kwargs = {k: v for k, v in kwargs.items() if k in ("chunk_size", "chunk_overlap")}

    chunker = get_chunker(strategy, **kwargs)
    return chunker.split(docs)


# ═══════════════════════════════════════════════════════════════════════════════
# Run Embedder
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Đang load embedding model...")  
def _load_hf_embedder_cached(
    model_name: str,
    device: str,
    trust_remote_code: bool,
    hf_token: str | None,
):
    """Load HuggingFaceEmbeddings một lần, cache theo key = tham số."""
    from langchain_huggingface import HuggingFaceEmbeddings

    model_kw: dict = {"device": device}
    if trust_remote_code:
        model_kw["trust_remote_code"] = True
    if hf_token:
        model_kw["token"] = hf_token
    if device == "cuda":
        try:
            import torch
            model_kw["model_kwargs"] = {"torch_dtype": torch.float16}
        except Exception:
            pass

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kw,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


_HF_TRUST_RC = frozenset({
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
})


def run_embedder(chunks: list, embed_params: dict) -> dict:
    """
    Chạy embedding trên toàn bộ chunks.
    Trả về: {"dense": list[list[float]], "dims": int, "n_embedded": int, "langchain_embedder": obj}
    """
    import math

    provider   = embed_params["provider"]
    model_name = embed_params["model_name"]
    device     = embed_params.get("device", "cpu")
    base_url   = embed_params.get("ollama_base_url", "http://localhost:11434")

    texts = [c.page_content for c in chunks]

    if provider == "huggingface":
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
        lc_embedder = _load_hf_embedder_cached(
            model_name=model_name,
            device=device,
            trust_remote_code=model_name in _HF_TRUST_RC,
            hf_token=hf_token or None,
        )
        dense = lc_embedder.embed_documents(texts)
        # Giải phóng CUDA fragment
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    elif provider == "ollama":
        from embedding.factory import get_embedder
        embedder = get_embedder("ollama", model_name, base_url=base_url)
        dense    = embedder.embed_documents(texts)
        lc_embedder = embedder.embedder
    else:
        raise ValueError(f"Provider không hỗ trợ: {provider}")

    # Sanitize NaN/Inf
    def _clean(v):
        return [0.0 if (x != x or not math.isfinite(x)) else x for x in v]
    dense = [_clean(v) for v in dense]

    dims = len(dense[0]) if dense else 0
    return {
        "dense":             dense,
        "dims":              dims,
        "n_embedded":        len(dense),
        "langchain_embedder": lc_embedder,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Run Vector DB (FAISS)
# ═══════════════════════════════════════════════════════════════════════════════

def run_vector_db_faiss(chunks: list, lc_embedder, vdb_params: dict):
    """
    Xây dựng hoặc load FAISS index.
    Trả về langchain FAISS VectorStore object.
    """
    from vector_db.faiss_store import FAISSVectorStore

    store = FAISSVectorStore(
        collection_name=vdb_params.get("collection_name", "rag_lab"),
        persist_dir=vdb_params.get("persist_dir", "./storage/faiss_rag_lab"),
        force_reindex=vdb_params.get("force_reindex", False),
    )
    return store.get_or_create(chunks, lc_embedder)



# ═══════════════════════════════════════════════════════════════════════════════
# Results: Loader
# ═══════════════════════════════════════════════════════════════════════════════

def render_loader_results(docs: list):
    """Hiển thị kết quả loader."""
    st.markdown("---")
    st.header("📂 Kết quả Loader")

    if not docs:
        st.warning("Không có document nào được tạo ra.")
        return

    total_chars = sum(len(d.page_content) for d in docs)
    file_types: dict[str, int] = {}
    for d in docs:
        ft = d.metadata.get("file_type", "unknown")
        file_types[ft] = file_types.get(ft, 0) + 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Tổng documents", len(docs))
    c2.metric("🔤 Tổng ký tự", f"{total_chars:,}")
    c3.metric("📊 Ký tự TB / doc", f"{total_chars // max(len(docs), 1):,}")
    c4.metric("📁 Loại file", len(file_types))

    if file_types:
        st.markdown("**Phân bổ loại file:**")
        fcols = st.columns(min(len(file_types), 6))
        for i, (ft, count) in enumerate(sorted(file_types.items())):
            fcols[i % len(fcols)].markdown(f"{_file_type_badge(ft)} **×{count}**", unsafe_allow_html=True)
        st.markdown("")

    search_q = st.text_input("🔍 Tìm kiếm trong nội dung", placeholder="Nhập từ khoá…", key="loader_search")

    filtered = [
        (i, doc) for i, doc in enumerate(docs)
        if not search_q or search_q.lower() in doc.page_content.lower()
    ]

    total_pages = max(1, (len(filtered) + _LOADER_PAGE_SIZE - 1) // _LOADER_PAGE_SIZE)
    if "loader_page" not in st.session_state or search_q != st.session_state.get("_loader_search_prev"):
        st.session_state["loader_page"] = 0
        st.session_state["_loader_search_prev"] = search_q
    page = st.session_state["loader_page"]

    if total_pages > 1:
        pc, pi, pn = st.columns([1, 3, 1])
        with pc:
            if st.button("◀ Trước", key="loader_prev", disabled=(page == 0)):
                st.session_state["loader_page"] = max(0, page - 1)
        with pi:
            st.caption(f"Trang {page+1}/{total_pages}  ·  {len(filtered)} kết quả")
        with pn:
            if st.button("Sau ▶", key="loader_next", disabled=(page >= total_pages - 1)):
                st.session_state["loader_page"] = min(total_pages - 1, page + 1)

    start = page * _LOADER_PAGE_SIZE
    for i, doc in filtered[start: start + _LOADER_PAGE_SIZE]:
        content  = doc.page_content
        meta     = doc.metadata
        ft       = meta.get("file_type", "unknown")
        source   = Path(meta.get("source", "")).name or "unknown"
        page_n   = meta.get("page", "")
        chars    = len(content)
        page_inf = f" · Trang {page_n}" if page_n else ""
        label    = f"Doc {i+1}  |  {source}{page_inf}  |  {chars:,} ký tự"

        with st.expander(label, expanded=(i == 0 and page == 0)):
            st.markdown(_file_type_badge(ft), unsafe_allow_html=True)
            st.markdown("")
            render_content_with_images(content)
            st.markdown("**Metadata:**")
            meta_clean = {k: v for k, v in meta.items() if v is not None and v != ""}
            st.json(meta_clean, expanded=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Results: Chunking
# ═══════════════════════════════════════════════════════════════════════════════

def render_chunking_results(chunks: list, strategy: str):
    """Hiển thị kết quả chunking."""
    st.markdown("---")
    st.header("✂️ Kết quả Chunking")

    if not chunks:
        st.warning("Không có chunk nào được tạo ra.")
        return

    sizes = [len(c.page_content) for c in chunks]
    total = sum(sizes)
    avg   = total // len(sizes)
    mn    = min(sizes)
    mx    = max(sizes)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🧩 Tổng chunks",  len(chunks))
    c2.metric("🔤 Tổng ký tự",   f"{total:,}")
    c3.metric("📊 TB / chunk",    f"{avg:,}")
    c4.metric("⬇️ Nhỏ nhất",     f"{mn:,}")
    c5.metric("⬆️ Lớn nhất",     f"{mx:,}")

    # Size distribution
    st.markdown("**Phân bổ kích thước chunk:**")
    buckets = {"<200": 0, "200-500": 0, "500-1000": 0, "1000-2000": 0, ">2000": 0}
    for s in sizes:
        if s < 200:    buckets["<200"]      += 1
        elif s < 500:  buckets["200-500"]    += 1
        elif s < 1000: buckets["500-1000"]   += 1
        elif s < 2000: buckets["1000-2000"]  += 1
        else:          buckets[">2000"]      += 1

    bcols = st.columns(5)
    for i, (lbl, cnt) in enumerate(buckets.items()):
        pct = cnt / len(chunks) * 100
        bcols[i].metric(lbl, f"{cnt} ({pct:.0f}%)")

    # Hierarchy info
    levels = set(c.metadata.get("chunk_level", "") for c in chunks)
    levels.discard("")
    if levels:
        st.info(f"🏗️ Hierarchical chunks: {', '.join(sorted(levels))}")

    # Filter
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        chunk_search = st.text_input("🔍 Tìm trong chunk", placeholder="Nhập từ khoá…", key="chunk_search")
    with col_f2:
        if levels:
            level_filter = st.selectbox("Lọc level", ["Tất cả"] + sorted(levels))
        else:
            level_filter = "Tất cả"

    filtered_chunks = [
        (i, c) for i, c in enumerate(chunks)
        if (not chunk_search or chunk_search.lower() in c.page_content.lower())
        and (level_filter == "Tất cả" or c.metadata.get("chunk_level", "") == level_filter)
    ]

    if not filtered_chunks:
        st.info("Không có chunk nào khớp với bộ lọc.")
        return

    total_pages = max(1, (len(filtered_chunks) + _CHUNKING_PAGE_SIZE - 1) // _CHUNKING_PAGE_SIZE)
    _prev_s = st.session_state.get("_chunk_search_prev")
    _prev_l = st.session_state.get("_chunk_level_prev")
    if chunk_search != _prev_s or level_filter != _prev_l:
        st.session_state["chunk_page"] = 0
        st.session_state["_chunk_search_prev"] = chunk_search
        st.session_state["_chunk_level_prev"]  = level_filter
    if "chunk_page" not in st.session_state:
        st.session_state["chunk_page"] = 0
    page = st.session_state["chunk_page"]

    if total_pages > 1:
        pc, pi, pn = st.columns([1, 3, 1])
        with pc:
            if st.button("◀ Trước", key="chunk_prev", disabled=(page == 0)):
                st.session_state["chunk_page"] = max(0, page - 1)
        with pi:
            st.caption(f"Trang {page+1}/{total_pages}  ·  {len(filtered_chunks)} kết quả")
        with pn:
            if st.button("Sau ▶", key="chunk_next", disabled=(page >= total_pages - 1)):
                st.session_state["chunk_page"] = min(total_pages - 1, page + 1)

    start = page * _CHUNKING_PAGE_SIZE
    for i, chunk in filtered_chunks[start: start + _CHUNKING_PAGE_SIZE]:
        content     = chunk.page_content
        meta        = chunk.metadata
        char_count  = len(content)
        source      = Path(meta.get("source", "")).name or ""
        chunk_level = meta.get("chunk_level", "")

        source_info = f" · {source}" if source else ""
        level_info  = f" · {chunk_level}" if chunk_level else ""
        label = f"Chunk {i+1}{level_info}{source_info}  |  {char_count:,} ký tự"

        with st.expander(label, expanded=(i < 3 and page == 0)):
            badge_html = _chunk_level_badge(chunk_level)
            if badge_html:
                st.markdown(badge_html, unsafe_allow_html=True)
                st.markdown("")
            render_content_with_images(content)
            meta_display = {k: v for k, v in meta.items() if v is not None and v != "" and k != "late_embedding"}
            st.markdown("**Metadata:**")
            st.json(meta_display, expanded=False)

    st.caption(f"Hiển thị {len(filtered_chunks[start:start+_CHUNKING_PAGE_SIZE])} / {len(filtered_chunks)} chunks (trang {page+1})")


# ═══════════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Loader & Chunking Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cache = PipelineCache("processed_data")

    # ── Global CSS ─────────────────────────────────────────────────────────
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 16px !important;
        }
        /* ── Header gradient ── */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            color: white;
        }
        .main-header h1 { color: white; font-size: 2.4rem; font-weight: 700; margin: 0; }
        .main-header p  { color: rgba(255,255,255,0.85); font-size: 1.1rem; margin: 0.5rem 0 0; }
        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        }
        section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
        /* ── Dropdown / Select: nền trắng → chữ đen ── */
        section[data-testid="stSidebar"] div[data-baseweb="select"] *,
        section[data-testid="stSidebar"] div[data-baseweb="popover"] *,
        section[data-testid="stSidebar"] li[role="option"],
        section[data-testid="stSidebar"] li[role="option"] *,
        section[data-testid="stSidebar"] div[data-baseweb="menu"] *,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] input::placeholder {
            color: #111827 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stCheckbox label {
            font-size: 14px !important; font-weight: 500 !important;
            color: #e0e0e0 !important;
        }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #a78bfa !important; font-size: 1rem !important;
        }
        /* ── Metric cards ── */
        [data-testid="stMetricValue"]  { font-size: 1.6rem !important; font-weight: 700 !important; }
        [data-testid="stMetricLabel"]  { font-size: 13px !important; }
        /* ── Expanders ── */
        .streamlit-expanderHeader,
        [data-testid="stExpander"] summary {
            font-size: 15px !important; font-weight: 600 !important;
            background: #f8f9ff !important; border-radius: 8px !important;
        }
        /* ── Buttons ── */
        .stButton button {
            font-size: 15px !important; font-weight: 600 !important;
            border-radius: 8px !important;
        }
        /* ── Upload area ── */
        [data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #f0f4ff, #e8f5ff) !important;
            border: 2px dashed #667eea !important;
            border-radius: 12px !important;
            padding: 1rem !important;
        }
        /* ── Code & JSON ── */
        .stMarkdown code {
            background: #fff3cd !important; color: #856404 !important;
            padding: 2px 7px !important; border-radius: 3px !important;
            font-size: 14px !important;
        }
        /* ── Tabs ── */
        [data-testid="stTabs"] [data-baseweb="tab"] {
            font-size: 15px !important; font-weight: 600 !important;
        }
        /* ── Progress bar ── */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Loader & Chunking Lab</h1>
        <p>Thử nghiệm tất cả kỹ thuật Loader và Chunking cho RAG pipeline · Upload file → Chọn chiến lược → Xem kết quả</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <div style="font-size:2.5rem;">🔬</div>
            <div style="font-size:1.1rem; font-weight:700; color:#a78bfa;">Loader & Chunking Lab</div>
            <div style="font-size:12px; color:#888; margin-top:4px;">v1.0.0</div>
        </div>
        <hr style="border-color:#333; margin: 0.5rem 0 1rem;">
        """, unsafe_allow_html=True)

        # (Đã loại bỏ Display mode toggle, mặc định dùng Markdown)

        st.markdown("<hr style='border-color:#333; margin:1rem 0;'>", unsafe_allow_html=True)

        loader_params = render_loader_settings()

        st.markdown("<hr style='border-color:#333; margin:1rem 0;'>", unsafe_allow_html=True)

        strategy, chunk_size, chunk_overlap, extra = render_chunking_settings()

        st.markdown("<hr style='border-color:#333; margin:1rem 0;'>", unsafe_allow_html=True)

        # Deduplication
        st.subheader("🔧 Hậu xử lý")
        do_dedup = st.checkbox("Deduplication chunks", value=False, help="Loại bỏ chunk trùng nội dung (MinHash LSH)")
        if do_dedup:
            dedup_method = st.selectbox("Phương pháp dedup", ["minhash", "exact"], key="dedup_method")
            dedup_threshold = st.slider("Ngưỡng similarity", 0.5, 1.0, 0.9, 0.05)
        else:
            dedup_method    = "minhash"
            dedup_threshold = 0.9

        st.markdown("<hr style='border-color:#333; margin:1rem 0;'>", unsafe_allow_html=True)
        st.subheader("🧠 Embedding")
        embed_model_opts = {
            "BAAI/bge-m3": {"provider": "huggingface", "display": "BGE-M3 (HuggingFace)", "dim": 1024},
            "qwen3-embedding:8b": {"provider": "ollama",      "display": "Qwen3-Embedding-8B (Ollama)", "dim": 4096},
        }
        embed_model_key = st.selectbox(
            "Embedding Model",
            options=list(embed_model_opts.keys()),
            format_func=lambda k: embed_model_opts[k]["display"],
            key="sel_embed_model",
        )
        embed_provider = embed_model_opts[embed_model_key]["provider"]
        embed_dim      = embed_model_opts[embed_model_key]["dim"]

        if embed_provider == "huggingface":
            _gpu_dev, _gpu_lbl = _detect_gpu()
            embed_device = st.radio(
                "Device",
                options=[_gpu_dev, "cpu"] if _gpu_dev != "cpu" else ["cpu"],
                index=0, horizontal=True, key="sel_embed_device",
            )
            if _gpu_dev == "cpu":
                st.caption("⚠️ Không phát hiện GPU — sẽ chạy trên CPU (chậm hơn)")
        else:
            embed_device = "cpu"  # Ollama tự quản lý device

        ollama_embed_url = "http://localhost:11434"
        if embed_provider == "ollama":
            ollama_embed_url = st.text_input(
                "Ollama base URL", value="http://localhost:11434", key="embed_ollama_url"
            )
            st.caption(f"💡 Pull model: `ollama pull {embed_model_key}`")

        embed_params = {
            "provider":       embed_provider,
            "model_name":     embed_model_key,
            "device":         embed_device,
            "ollama_base_url": ollama_embed_url,
            "dim":            embed_dim,
        }

        st.markdown("<hr style='border-color:#333; margin:1rem 0;'>", unsafe_allow_html=True)
        st.subheader("🗄️ Vector DB (FAISS)")
        vdb_collection = st.text_input(
            "Collection name", value="rag_lab", key="vdb_collection",
            help="Tên collection FAISS — ảnh hưởng đến thư mục lưu index"
        )
        vdb_persist = st.text_input(
            "Persist dir", value=f"./storage/faiss_{vdb_collection}", key="vdb_persist"
        )
        vdb_force = st.checkbox("Force re-index", value=False, key="vdb_force",
                                help="Xóa index cũ và build lại dù corpus không đổi")

        vdb_params = {
            "collection_name": vdb_collection,
            "persist_dir":     vdb_persist,
            "force_reindex":   vdb_force,
        }

        st.markdown("<hr style='border-color:#333; margin:1rem 0;'>", unsafe_allow_html=True)
        st.subheader("💾 Bộ nhớ Cache")
        try:
            cache_size = cache.total_size_mb()
            st.markdown(f"Dung lượng: **{cache_size:.2f} MB**")
        except Exception:
            st.markdown("Dung lượng: **0.00 MB**")
        
        if st.button("🗑️ Clear Cache", use_container_width=True, help="Xóa toàn bộ cache đã lưu trên disk"):
            cache.clear_all()
            st.success("Đã xóa sạch bộ nhớ cache!")
            time.sleep(0.5)
            st.rerun()

    # ── Main Content ────────────────────────────────────────────────────────
    tab_upload, tab_results, tab_embed, tab_compare, tab_about = st.tabs([
        "📁 Upload & Process", "📊 Kết quả", "🧠 Embedding & VectorDB", "⚖️ So sánh", "ℹ️ Hướng dẫn"
    ])

    # ─── Tab 1: Upload & Process ───────────────────────────────────────────
    with tab_upload:
        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.subheader("📁 Upload File")
            uploaded_files = st.file_uploader(
                "Kéo & thả file PDF vào đây",
                type=["pdf"],
                accept_multiple_files=True,
                help="Hỗ trợ PDF. Có thể upload nhiều file cùng lúc.",
                key="file_uploader",
            )

            if uploaded_files:
                st.markdown(f"**{len(uploaded_files)} file đã upload:**")
                for f in uploaded_files:
                    size_kb = f.size / 1024
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;padding:6px 12px;'
                        f'background:#f0f4ff;border-radius:8px;margin:4px 0;">'
                        f'<span style="font-size:20px;">📄</span>'
                        f'<span style="font-weight:600;">{f.name}</span>'
                        f'<span style="color:#888;font-size:13px;margin-left:auto;">{size_kb:.1f} KB</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with col_right:
            st.subheader("⚙️ Cấu hình hiện tại")
            st.markdown(f"""
            <div style="background:#f8f9ff;border-radius:12px;padding:1rem 1.5rem;">
                <div style="margin-bottom:8px;">
                    <span style="color:#667eea;font-weight:700;">📥 Loader</span><br>
                    <code style="font-size:15px;">{loader_params['pdf_strategy']}</code>
                    <span style="color:#888;font-size:13px;"> · {loader_params['language']}</span>
                </div>
                <div>
                    <span style="color:#764ba2;font-weight:700;">✂️ Chunker</span><br>
                    <code style="font-size:15px;">{strategy}</code>
                    <span style="color:#888;font-size:13px;"> · size={chunk_size} · overlap={chunk_overlap}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # ── Process button ────────────────────────────────────────────
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                run_loader_only = st.button(
                    "📥 Chỉ chạy Loader",
                    type="secondary",
                    use_container_width=True,
                    disabled=not uploaded_files,
                    help="Chỉ chạy bước loading, chưa chunking",
                )
            with col_btn2:
                run_all = st.button(
                    "▶️ Loader + Chunking",
                    type="primary",
                    use_container_width=True,
                    disabled=not uploaded_files,
                    help="Chạy cả hai bước: loader và chunking",
                )
            with col_btn3:
                run_full = st.button(
                    "🚀 Chạy Toàn Bộ",
                    type="primary",
                    use_container_width=True,
                    disabled=not uploaded_files,
                    help="Chạy Loader + Chunking + Embedding + Vector DB",
                )

        # ── Execute ──────────────────────────────────────────────────────
        if (run_all or run_loader_only or run_full) and uploaded_files:
            # Lưu file tạm
            tmp_paths = []
            tmp_dir_obj = tempfile.TemporaryDirectory()
            tmp_dir = tmp_dir_obj.name

            for f in uploaded_files:
                tmp_path = Path(tmp_dir) / f.name
                tmp_path.write_bytes(f.read())
                tmp_paths.append(str(tmp_path))

            # Tính toán cache keys
            source_path_str = ",".join(tmp_paths)
            input_display_str = ", ".join(f.name for f in uploaded_files)
            input_hash = cache.compute_input_hash(source_path_str)
            st.session_state["_last_input_hash"] = input_hash

            _LOADER_SKIP_CACHE_KEYS = {"ollama_base_url"}
            loader_cfg = {k: v for k, v in loader_params.items() if k not in _LOADER_SKIP_CACHE_KEYS}
            loader_key = cache.make_step_key(input_hash, loader_cfg)

            # ── Run Loader ──────────────────────────────────────────────
            cached_docs = cache.load_loader(input_hash, loader_key)
            if cached_docs is not None:
                all_docs = cached_docs
                st.info(f"⚡ [Cache HIT] Đã tải **{len(all_docs)}** documents từ cache (tiết kiệm thời gian)!")
                st.session_state["_docs"] = all_docs
                st.session_state["_loader_params"] = loader_params
            else:
                with st.spinner(f"⏳ Đang chạy loader **{loader_params['pdf_strategy']}**…"):
                    try:
                        all_docs = []
                        progress = st.progress(0, text="Loading files…")
                        for idx, p in enumerate(tmp_paths):
                            docs = run_loader(p, loader_params)
                            all_docs.extend(docs)
                            progress.progress((idx + 1) / len(tmp_paths), text=f"Đã load {idx+1}/{len(tmp_paths)} file")
                        progress.empty()

                        # Lưu vào cache
                        cache.save_loader(input_hash, loader_key, all_docs, loader_cfg, source_path=source_path_str, input_display=input_display_str)

                        st.session_state["_docs"] = all_docs
                        st.session_state["_loader_params"] = loader_params
                        st.success(f"✅ Loader hoàn tất: **{len(all_docs)}** documents từ {len(tmp_paths)} file")
                    except Exception as e:
                        st.error(f"❌ Loader thất bại: {e}")
                        st.exception(e)
                        tmp_dir_obj.cleanup()
                        st.stop()

            # ── Step 2: Chunking ─────────────────────────────────────────
            if run_all or run_full:
                all_docs = st.session_state.get("_docs", [])
                if all_docs:
                    st.markdown("""
                    <div style='display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,#1e3a5f,#1a2e4a);
                         border-radius:10px;padding:.75rem 1.2rem;margin:.5rem 0;border-left:4px solid #60a5fa;'>
                        <span style='font-size:1.4rem;'>✂️</span>
                        <div>
                            <div style='color:#93c5fd;font-weight:700;font-size:15px;'>Bước 2 — Chunking</div>
                            <div style='color:#94a3b8;font-size:12px;'>Đang chia nhỏ documents thành chunks...</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    _CHUNK_SKIP_CACHE_KEYS = {"ollama_base_url"}
                    chunking_cfg = {
                        "strategy": strategy,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        **{k: v for k, v in extra.items() if k not in _CHUNK_SKIP_CACHE_KEYS}
                    }
                    chunking_cfg_for_key = dict(chunking_cfg)
                    chunking_cfg_for_key["do_dedup"] = do_dedup
                    if do_dedup:
                        chunking_cfg_for_key["dedup_method"] = dedup_method
                        chunking_cfg_for_key["dedup_threshold"] = dedup_threshold

                    chunk_key = cache.make_step_key(loader_key, chunking_cfg_for_key)
                    cached_chunks = cache.load_chunking(input_hash, chunk_key)

                    if cached_chunks is not None:
                        chunks = cached_chunks
                        st.info(f"⚡ [Cache HIT] Đã tải **{len(chunks)}** chunks từ cache!")
                        st.session_state["_chunks"] = chunks
                        st.session_state["_chunk_strategy"] = strategy
                        st.session_state["_last_chunk_key"] = chunk_key
                    else:
                        with st.spinner(f"⏳ Đang chạy chunker **{strategy}**…"):
                            try:
                                chunks = run_chunker(all_docs, strategy, chunk_size, chunk_overlap, extra)

                                # Deduplication
                                if do_dedup and chunks:
                                    from chunking.deduplication import deduplicate_chunks
                                    before = len(chunks)
                                    chunks = deduplicate_chunks(chunks, method=dedup_method, threshold=dedup_threshold)
                                    after  = len(chunks)
                                    if before != after:
                                        st.info(f"🔁 Dedup: {before} → {after} chunks (loại bỏ {before - after} chunk trùng)")

                                # Lưu vào cache
                                cache.save_chunking(input_hash, chunk_key, chunks, chunking_cfg_for_key, loader_key=loader_key)

                                st.session_state["_chunks"] = chunks
                                st.session_state["_chunk_strategy"] = strategy
                                st.session_state["_last_chunk_key"] = chunk_key
                                st.success(f"✅ Chunking hoàn tất: **{len(chunks)}** chunks")
                            except Exception as e:
                                st.error(f"❌ Chunking thất bại: {e}")
                                st.exception(e)
                                st.stop()

            # ── Step 3+4: Embedding + Vector DB (chỉ khi run_full) ─────
            if run_full:
                chunks = st.session_state.get("_chunks", [])
                chunk_key = st.session_state.get("_last_chunk_key", "")
                if not chunks:
                    st.warning("⚠️ Chưa có chunks — không thể chạy Embedding. Hãy thử lại.")
                else:
                    # ── Bước 3: Embedding ────────────────────────────────
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,#2d1b69,#1e1240);
                         border-radius:10px;padding:.75rem 1.2rem;margin:.5rem 0;border-left:4px solid #a78bfa;'>
                        <span style='font-size:1.4rem;'>🧠</span>
                        <div>
                            <div style='color:#c4b5fd;font-weight:700;font-size:15px;'>Bước 3 — Embedding</div>
                            <div style='color:#94a3b8;font-size:12px;'>Model: {embed_params['model_name']} · {len(chunks)} chunks sẽ được vector hoá</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    _embed_cfg_for_key = {k: v for k, v in embed_params.items() if k not in {"ollama_base_url", "dim"}}
                    _embed_key = cache.make_step_key(chunk_key, _embed_cfg_for_key) if chunk_key else cache.make_step_key(input_hash, _embed_cfg_for_key)
                    _cached_embed = cache.load_embedding(input_hash, _embed_key) if input_hash else None

                    if _cached_embed is not None:
                        embed_result = _cached_embed
                        if embed_params["provider"] == "huggingface":
                            lc_embedder = _load_hf_embedder_cached(
                                model_name=embed_params["model_name"],
                                device=embed_params["device"],
                                trust_remote_code=embed_params["model_name"] in _HF_TRUST_RC,
                                hf_token=os.environ.get("HF_TOKEN") or None,
                            )
                        else:
                            from embedding.factory import get_embedder as _get_emb
                            lc_embedder = _get_emb(
                                "ollama", embed_params["model_name"],
                                base_url=embed_params["ollama_base_url"]
                            ).embedder
                        st.info(f"⚡ [Embedding Cache HIT] Sử dụng {embed_result['n_embedded']} vectors đã cache.")
                    else:
                        with st.spinner(f"⏳ Đang tính embedding bằng **{embed_params['model_name']}** ({len(chunks)} chunks)..."):
                            embed_progress = st.progress(0, text="Chuẩn bị model embedding…")
                            try:
                                embed_progress.progress(10, text="Đang load model…")
                                embed_result_full = run_embedder(chunks, embed_params)
                                embed_progress.progress(90, text="Sanitize vectors…")
                                dense       = embed_result_full["dense"]
                                lc_embedder = embed_result_full["langchain_embedder"]
                                embed_result = {
                                    "dense":      dense,
                                    "sparse":     None,
                                    "dims":       embed_result_full["dims"],
                                    "n_embedded": embed_result_full["n_embedded"],
                                    "truncated":  False,
                                }
                                if input_hash:
                                    cache.save_embedding(input_hash, _embed_key, embed_result, _embed_cfg_for_key, chunk_key=chunk_key)
                                embed_progress.progress(100, text="Embedding xong!")
                                embed_progress.empty()
                                st.success(f"✅ Embedding hoàn tất: **{embed_result['n_embedded']}** vectors · dim={embed_result['dims']}")
                            except Exception as e:
                                embed_progress.empty()
                                st.error(f"❌ Embedding thất bại: {e}")
                                st.exception(e)
                                st.stop()

                    # ── Bước 4: Vector DB (FAISS) ────────────────────────
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,#14532d,#0d3320);
                         border-radius:10px;padding:.75rem 1.2rem;margin:.5rem 0;border-left:4px solid #4ade80;'>
                        <span style='font-size:1.4rem;'>🗄️</span>
                        <div>
                            <div style='color:#86efac;font-weight:700;font-size:15px;'>Bước 4 — Vector DB (FAISS)</div>
                            <div style='color:#94a3b8;font-size:12px;'>Collection: {vdb_params['collection_name']} · Lưu tại: {vdb_params['persist_dir']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.spinner("⏳ Đang xây dựng FAISS index..."):
                        faiss_progress = st.progress(0, text="Khởi tạo FAISS index…")
                        try:
                            faiss_progress.progress(30, text="Đang index vectors…")
                            faiss_store = run_vector_db_faiss(chunks, lc_embedder, vdb_params)
                            faiss_progress.progress(80, text="Lưu index xuống disk…")
                            st.session_state["_faiss_store"]  = faiss_store
                            st.session_state["_embed_result"] = embed_result
                            st.session_state["_embed_params"] = embed_params
                            st.session_state["_vdb_params"]   = vdb_params
                            st.session_state["_lc_embedder"]  = lc_embedder
                            faiss_progress.progress(100, text="Hoàn tất!")
                            faiss_progress.empty()
                            st.success("🎉 **Toàn bộ pipeline đã hoàn tất!** Bạn có thể chuyển sang Tab **🧠 Embedding & VectorDB** để test truy vấn.")
                        except Exception as e:
                            faiss_progress.empty()
                            st.error(f"❌ FAISS thất bại: {e}")
                            st.exception(e)

            tmp_dir_obj.cleanup()

        # ── Status ───────────────────────────────────────────────────────
        docs   = st.session_state.get("_docs",   [])
        chunks = st.session_state.get("_chunks", [])

        if docs or chunks:
            st.markdown("---")
            st.markdown("**📊 Trạng thái session:**")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("📄 Documents loaded",  len(docs))
            sc2.metric("🧩 Chunks created",    len(chunks))
            sc3.metric("📥 Loader strategy",   st.session_state.get("_loader_params", {}).get("pdf_strategy", "—"))

            if st.button("🗑️ Xoá kết quả", type="secondary"):
                for k in ("_docs", "_chunks", "_loader_params", "_chunk_strategy"):
                    st.session_state.pop(k, None)
                st.rerun()

    # ─── Tab 2: Results ─────────────────────────────────────────────────────
    with tab_results:
        docs   = st.session_state.get("_docs",   [])
        chunks = st.session_state.get("_chunks", [])

        if not docs and not chunks:
            st.info("💡 Upload file và chạy ở tab **Upload & Process** để xem kết quả tại đây.")
        else:
            if docs:
                render_loader_results(docs)
            if chunks:
                render_chunking_results(
                    chunks,
                    strategy=st.session_state.get("_chunk_strategy", "")
                )

    # ─── Tab 3: Embedding & VectorDB ────────────────────────────────────────
    with tab_embed:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:1.5rem 2rem;
             border-radius:14px;margin-bottom:1.5rem;border:1px solid #333;'>
            <h2 style='color:#a78bfa;margin:0 0 6px;font-size:1.5rem;'>🧠 Embedding & Vector DB</h2>
            <p style='color:#aaa;margin:0;font-size:14px;'>Xem kết quả vector hoá và thử tìm kiếm tương đồng trên FAISS index.</p>
        </div>
        """, unsafe_allow_html=True)

        chunks_for_embed = st.session_state.get("_chunks", [])
        faiss_store_obj  = st.session_state.get("_faiss_store")

        # ── Thử load FAISS từ disk nếu session bị reset ──────────────────
        if faiss_store_obj is None and chunks_for_embed:
            _vdb_p = st.session_state.get("_vdb_params", vdb_params)
            _ep    = st.session_state.get("_embed_params", embed_params)
            from pathlib import Path as _P
            _idx = _P(_vdb_p.get("persist_dir", vdb_params["persist_dir"]))
            if (_idx / "index.faiss").exists():
                try:
                    with st.spinner("⏳ Đang load lại FAISS index từ disk..."):
                        if _ep["provider"] == "huggingface":
                            _lc = _load_hf_embedder_cached(
                                model_name=_ep["model_name"],
                                device=_ep.get("device", "cpu"),
                                trust_remote_code=_ep["model_name"] in _HF_TRUST_RC,
                                hf_token=os.environ.get("HF_TOKEN") or None,
                            )
                        else:
                            from embedding.factory import get_embedder as _ge
                            _lc = _ge("ollama", _ep["model_name"],
                                      base_url=_ep.get("ollama_base_url", "http://localhost:11434")).embedder
                        from langchain_community.vectorstores import FAISS as _FAISS
                        faiss_store_obj = _FAISS.load_local(
                            folder_path=str(_idx),
                            embeddings=_lc,
                            allow_dangerous_deserialization=True,
                        )
                        st.session_state["_faiss_store"]  = faiss_store_obj
                        st.session_state["_lc_embedder"]  = _lc
                        st.session_state["_embed_params"]  = _ep
                        st.session_state["_vdb_params"]    = _vdb_p
                    st.info(f"⚡ FAISS index được load lại từ `{_idx}`.")
                except Exception as _e:
                    st.warning(f"⚠️ Không load được FAISS từ disk: {_e}")

        if not chunks_for_embed:
            st.warning("⚠️ Chưa có chunks. Vui lòng chạy **Loader + Chunking** ở tab Upload & Process trước.")
        else:
            st.info(f"📂 Sử dụng **{len(chunks_for_embed)}** chunks từ bước Chunking.")

            # ── Config cards ──────────────────────────────────────────────
            ec1, ec2 = st.columns(2)
            _ep_show = st.session_state.get("_embed_params", embed_params)
            _vp_show = st.session_state.get("_vdb_params",   vdb_params)
            with ec1:
                st.markdown(f"""
                <div style='background:#f8f9ff;border-radius:10px;padding:.9rem 1.2rem;'>
                    <div style='color:#667eea;font-weight:700;margin-bottom:4px;'>🧠 Embedding Model</div>
                    <code style='font-size:15px;color:#111;'>{_ep_show.get('model_name','—')}</code>
                    <div style='color:#888;font-size:12px;margin-top:4px;'>Provider: {_ep_show.get('provider','—')} · Dim: {_ep_show.get('dim','—')}</div>
                </div>
                """, unsafe_allow_html=True)
            with ec2:
                st.markdown(f"""
                <div style='background:#f8f9ff;border-radius:10px;padding:.9rem 1.2rem;'>
                    <div style='color:#764ba2;font-weight:700;margin-bottom:4px;'>🗄️ Vector DB</div>
                    <code style='font-size:15px;color:#111;'>FAISS</code>
                    <div style='color:#888;font-size:12px;margin-top:4px;'>Collection: {_vp_show.get('collection_name','—')} · Dir: {_vp_show.get('persist_dir','—')}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("")

            # ── Nút chạy nếu chưa có index ───────────────────────────────
            if faiss_store_obj is None:
                st.warning("⚠️ Chưa có FAISS index. Bấm **Chạy Embedding + Index FAISS** hoặc dùng nút **🚀 Chạy Toàn Bộ** ở tab Upload & Process.")
                if st.button("🚀 Chạy Embedding + Index FAISS", type="primary", use_container_width=True, key="btn_run_embed_tab"):
                    _embed_cfg_for_key = {k: v for k, v in embed_params.items() if k not in {"ollama_base_url", "dim"}}
                    _input_hash_embed  = st.session_state.get("_last_input_hash", "")
                    _chunk_key_embed   = st.session_state.get("_last_chunk_key", "")
                    _embed_key         = cache.make_step_key(_chunk_key_embed or _input_hash_embed or "default", _embed_cfg_for_key)
                    _cached_embed      = cache.load_embedding(_input_hash_embed, _embed_key) if _input_hash_embed else None

                    if _cached_embed is not None:
                        embed_result_tab = _cached_embed
                        if embed_params["provider"] == "huggingface":
                            lc_emb = _load_hf_embedder_cached(
                                model_name=embed_params["model_name"], device=embed_params.get("device","cpu"),
                                trust_remote_code=embed_params["model_name"] in _HF_TRUST_RC,
                                hf_token=os.environ.get("HF_TOKEN") or None,
                            )
                        else:
                            from embedding.factory import get_embedder as _ge2
                            lc_emb = _ge2("ollama", embed_params["model_name"], base_url=embed_params.get("ollama_base_url","http://localhost:11434")).embedder
                        st.info(f"⚡ Cache HIT: {embed_result_tab['n_embedded']} vectors.")
                    else:
                        with st.spinner(f"⏳ Embedding với {embed_params['model_name']}..."):
                            try:
                                ef = run_embedder(chunks_for_embed, embed_params)
                                lc_emb = ef["langchain_embedder"]
                                embed_result_tab = {"dense": ef["dense"], "sparse": None, "dims": ef["dims"], "n_embedded": ef["n_embedded"], "truncated": False}
                                if _input_hash_embed:
                                    cache.save_embedding(_input_hash_embed, _embed_key, embed_result_tab, _embed_cfg_for_key, chunk_key=_chunk_key_embed)
                                st.success(f"✅ Embedding: {ef['n_embedded']} vectors · dim={ef['dims']}")
                            except Exception as e:
                                st.error(f"❌ Embedding thất bại: {e}"); st.stop()
                    with st.spinner("⏳ Đang xây dựng FAISS index..."):
                        try:
                            faiss_store_obj = run_vector_db_faiss(chunks_for_embed, lc_emb, vdb_params)
                            st.session_state["_faiss_store"]  = faiss_store_obj
                            st.session_state["_embed_result"] = embed_result_tab
                            st.session_state["_embed_params"] = embed_params
                            st.session_state["_vdb_params"]   = vdb_params
                            st.session_state["_lc_embedder"]  = lc_emb
                            st.success("✅ FAISS index sẵn sàng!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ FAISS thất bại: {e}")

            # ── Metrics + Search (hiển thị khi đã có index) ───────────────
            if faiss_store_obj is not None:
                st.markdown("---")
                st.subheader("📊 Thông tin Index")
                er = st.session_state.get("_embed_result", {})
                _vp2 = st.session_state.get("_vdb_params", vdb_params)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🧠 Vectors",    er.get("n_embedded", len(chunks_for_embed)))
                m2.metric("📐 Dimensions", er.get("dims", _ep_show.get("dim", "—")))
                m3.metric("🗄️ Provider",   _ep_show.get("provider", "—"))
                from pathlib import Path as _P2
                _ip = _P2(_vp2.get("persist_dir", ""))
                _sz = round(sum(f.stat().st_size for f in _ip.rglob("*") if f.is_file()) / 1_048_576, 3) if _ip.exists() else 0
                m4.metric("📁 Index Size", f"{_sz:.3f} MB")

                st.markdown("---")
                st.subheader("🔍 Test Similarity Search")
                query_txt = st.text_input(
                    "Nhập câu truy vấn:",
                    placeholder="Nhập câu hỏi bằng tiếng Việt hoặc tiếng Anh...",
                    key="embed_search_query"
                )
                top_k = st.slider("Top-K kết quả", 1, 10, 3, key="embed_top_k")

                if st.button("🔍 Search", key="btn_embed_search") and query_txt.strip():
                    with st.spinner("⏳ Đang tìm kiếm..."):
                        try:
                            results = faiss_store_obj.similarity_search(query_txt, k=top_k)
                            st.markdown(f"**{len(results)} kết quả gần nhất:**")
                            for i, doc in enumerate(results):
                                src  = Path(doc.metadata.get("source", "")).name or "?"
                                pg   = doc.metadata.get("page", "")
                                info = f" · Trang {pg}" if pg else ""
                                with st.expander(f"#{i+1}  {src}{info}  —  {len(doc.page_content):,} ký tự", expanded=(i == 0)):
                                    render_content_with_images(doc.page_content)
                                    st.json({k: v for k, v in doc.metadata.items() if v}, expanded=False)
                        except Exception as e:
                            st.error(f"❌ Search thất bại: {e}")

    # ─── Tab 4: So sánh ─────────────────────────────────────────────────────
    with tab_compare:
        st.subheader("⚖️ So sánh các kỹ thuật Loader & Chunking")


        st.markdown("""
        ### 📥 So sánh Loader

        | Strategy | Tốc độ | Chất lượng | OCR | Bảng | Ảnh | GPU |
        |---|---|---|---|---|---|---|
        | **pypdf** | ⚡⚡⚡ | ⭐⭐ | ❌ | ❌ | ❌ | ❌ |
        | **pymupdf** | ⚡⚡⚡ | ⭐⭐⭐ | ❌ | ❌ | ❌ | ❌ |
        | **pdfplumber** | ⚡⚡ | ⭐⭐⭐ | ❌ | ✅ tốt nhất | ❌ | ❌ |
        | **unstructured** | ⚡ | ⭐⭐⭐⭐ | ✅ | ✅ | ✅ | Optional |
        | **docling** | ⚡ | ⭐⭐⭐⭐⭐ | ✅ RapidOCR | ✅ | ✅ | Optional |
        | **marker** | ⚡ | ⭐⭐⭐⭐⭐ | ✅ Surya 90+ lang | ✅ | ✅ | Recommended |
        | **opendataloader** | ⚡⚡ | ⭐⭐⭐⭐⭐ (#1) | ✅ hybrid | ✅ 0.93 | ✅ | ❌ (Java) |

        ---

        ### ✂️ So sánh Chunking

        | Strategy | Mô tả | Tốt cho | LLM cần |
        |---|---|---|---|
        | **recursive** | Cắt theo paragraph → line → sentence | Mặc định, tổng quát | ❌ |
        | **token_based** | Đếm token BPE thay vì ký tự | Multilingual, strict token limit | ❌ |
        | **format_aware** | Cắt theo Markdown heading / code / HTML | Docs có cấu trúc rõ | ❌ |
        | **sentence_aware** | Cắt tại ranh giới câu | Q&A, FAQ, narrative text | ❌ |
        | **semantic** | Cắt theo cosine similarity | Docs nhiều chủ đề khác nhau | ❌ (embedding) |
        | **hierarchical** | Parent (lớn) + Child (nhỏ) | Docs dài, cần cả precision & context | ❌ |
        | **contextual** | Recursive + LLM context prefix | Khi recall là ưu tiên số 1 | ✅ |

        ---

        ### 💡 Gợi ý cấu hình theo use case

        | Use case | Loader | Chunker |
        |---|---|---|
        | PDF digital (text layer) | pypdf / pymupdf | recursive |
        | PDF scan (tiếng Việt) | marker (cuda) | recursive / sentence_aware |
        | Báo cáo tài chính (nhiều bảng) | pdfplumber / docling | format_aware (markdown) |
        | Research paper phức tạp | docling / marker | hierarchical |
        | FAQ / Q&A dataset | pypdf | sentence_aware |
        | Corpus đa chủ đề | docling | semantic |
        | RAG chất lượng cao (budget có) | marker | contextual |
        """)

    # ─── Tab 4: About ─────────────────────────────────────────────────────
    with tab_about:
        st.subheader("ℹ️ Hướng dẫn sử dụng")

        st.markdown("""
        ## 🚀 Bắt đầu nhanh

        ### Bước 1: Upload file
        - Vào tab **Upload & Process**
        - Kéo & thả file PDF vào vùng upload (hỗ trợ nhiều file cùng lúc)

        ### Bước 2: Chọn cấu hình (sidebar bên trái)
        - **Loader**: chọn PDF strategy phù hợp
        - **Chunking**: chọn strategy và cấu hình tham số
        - **Hiển thị**: chọn Text hoặc Markdown

        ### Bước 3: Chạy
        - **📥 Chỉ chạy Loader**: xem nội dung đã extract, chưa chunk
        - **▶️ Loader + Chunking**: chạy cả hai bước

        ### Bước 4: Xem kết quả
        - Vào tab **📊 Kết quả** để xem documents và chunks
        - Dùng thanh tìm kiếm để lọc nội dung
        - Xem Metadata của từng document/chunk

        ---

        ## 📦 Cài đặt thư viện

        ### Cài đặt cơ bản (đã có trong requirements.txt):
        ```bash
        pip install -r requirements.txt
        ```

        ### Loader nâng cao (cài thêm nếu cần):
        ```bash
        # Docling (IBM Research)
        pip install docling

        # Marker (high-quality Markdown)
        pip install marker-pdf

        # Unstructured (OCR + tables + figures)
        pip install "unstructured[pdf]" unstructured-inference

        # OpenDataLoader (#1 benchmark, cần Java 11+)
        pip install opendataloader-pdf

        # PyTorch với CUDA 12.8 (cho Marker)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        ```

        ---

        ## ⚙️ Chunking chiến lược đặc biệt

        ### Semantic Chunking
        Cần model embedding. Ví dụ dùng HuggingFace local:
        ```bash
        pip install sentence-transformers langchain-huggingface
        ```

        ### Contextual Chunking
        Cần LLM API key trong file `.env`:
        ```
        OPENAI_API_KEY=sk-...
        ANTHROPIC_API_KEY=sk-ant-...
        GOOGLE_API_KEY=AI...
        ```

        ---

        ## 📁 Cấu trúc project
        ```
        loader-chunking-lab/
        ├── app.py               ← Entry point (file này)
        ├── requirements.txt     ← Dependencies
        ├── loader/
        │   ├── base.py          ← Abstract base class
        │   ├── utils.py         ← clean_text, content_hash, table_html_to_markdown
        │   ├── pdf_loader.py    ← 7 PDF loader strategies
        │   └── directory_loader.py ← PDFDocumentLoader (dispatch)
        └── chunking/
            ├── base.py          ← Abstract base class
            ├── factory.py       ← get_chunker() factory
            ├── recursive.py     ← RecursiveChunker
            ├── token_based.py   ← TokenChunker
            ├── format_aware.py  ← FormatAwareChunker
            ├── sentence_aware.py ← SentenceChunker
            ├── semantic.py      ← SemanticChunker
            ├── hierarchical.py  ← HierarchicalChunker
            ├── contextual.py    ← ContextualChunker
            ├── deduplication.py ← deduplicate_chunks (MinHash)
            └── utils.py         ← call_llm, ensure_ollama_model
        ```
        """)


if __name__ == "__main__":
    main()
