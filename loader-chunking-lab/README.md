# Loader & Chunking Lab

Ứng dụng Streamlit độc lập để thử nghiệm tất cả các kỹ thuật **Loader** và **Chunking** cho RAG pipeline.

## 🚀 Khởi động nhanh

```bash
# Kích hoạt venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

# Cài thư viện
pip install -r requirements.txt

# Chạy app
streamlit run app.py
```

## 📦 Cấu trúc project

```
loader-chunking-lab/
├── app.py                   ← Entry point chính (Streamlit UI)
├── requirements.txt         ← Dependencies
├── .env.example             ← Mẫu file API key
├── lenh.txt                 ← Các lệnh thường dùng
├── loader/
│   ├── base.py              ← Abstract BaseLoader
│   ├── utils.py             ← clean_text, content_hash, table_html_to_markdown
│   ├── pdf_loader.py        ← 7 PDF loader strategies
│   └── directory_loader.py  ← PDFDocumentLoader dispatcher
└── chunking/
    ├── base.py              ← Abstract BaseChunker
    ├── factory.py           ← get_chunker() factory
    ├── recursive.py         ← RecursiveChunker
    ├── token_based.py       ← TokenChunker
    ├── format_aware.py      ← FormatAwareChunker
    ├── sentence_aware.py    ← SentenceChunker
    ├── semantic.py          ← SemanticChunker + EMBEDDING_MODELS registry
    ├── hierarchical.py      ← HierarchicalChunker
    ├── contextual.py        ← ContextualChunker
    ├── deduplication.py     ← deduplicate_chunks (MinHash LSH)
    └── utils.py             ← call_llm, ensure_ollama_model
```

## 📥 Loader Strategies

| Strategy | Package | Điểm mạnh |
|---|---|---|
| `pypdf` | pypdf (built-in) | Nhanh, text layer |
| `pymupdf` | `pip install pymupdf` | Nhanh nhất, layout tốt |
| `pdfplumber` | `pip install pdfplumber` | Bảng tốt nhất (text layer) |
| `unstructured` | `pip install "unstructured[pdf]"` | OCR + bảng + ảnh |
| `docling` | `pip install docling` | IBM, Markdown xuất sắc |
| `marker` | `pip install marker-pdf` | Markdown + LaTeX, 90+ ngôn ngữ |
| `opendataloader` | `pip install opendataloader-pdf` | #1 benchmark (accuracy 0.90), Java 11+ |

## ✂️ Chunking Strategies

| Strategy | Mô tả |
|---|---|
| `recursive` | Cắt theo đoạn → dòng → câu → ký tự (mặc định) |
| `token_based` | Đếm token BPE (tiktoken) |
| `format_aware` | Cắt theo Markdown heading / code / HTML |
| `sentence_aware` | Cắt tại ranh giới câu |
| `semantic` | Cắt theo cosine similarity (embedding) |
| `hierarchical` | Parent (lớn) + Child (nhỏ) |
| `contextual` | Recursive + LLM context prefix (Anthropic method) |

## 🎯 Tính năng

- 📁 **Upload nhiều file PDF** cùng lúc
- ⚙️ **Cấu hình đầy đủ** cho cả loader và chunker
- 📊 **Xem kết quả chi tiết**: thống kê, pagination, tìm kiếm
- 🖼️ **Render ảnh** từ Marker/Docling inline
- 🔁 **Deduplication** (MinHash LSH / exact hash)
- 📖 **Tab so sánh** các kỹ thuật

## ⚙️ Cài đặt bổ sung

### PyTorch với CUDA 12.8 (cho Marker với GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Loader nâng cao
```bash
pip install docling
pip install marker-pdf
pip install "unstructured[pdf]" unstructured-inference
pip install opendataloader-pdf   # Cần Java 11+ từ https://adoptium.net/
```

### API Keys (tạo file `.env`)
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
```
