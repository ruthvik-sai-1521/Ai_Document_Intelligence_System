# 🛠️ Developer & Engineering Guide

This guide provides instructions for setting up your local development environment, running test suites, writing custom document parsers, extending enterprise connectors, and maintaining static analysis rules.

---

## 💻 1. Local Development Setup

### System Prerequisites
- Python 3.10 installed.
- Git, C++ compiler (for building native libraries if needed).
- System libraries: `tesseract-ocr`, `poppler-utils`, `ffmpeg`.

### Environment Initialization
```bash
# Clone the repository
git clone https://github.com/your-org/Ai_Document_Intelligence_System.git
cd Ai_Document_Intelligence_System

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies in editable mode
pip install --upgrade pip setuptools wheel
pip install PyJWT sentence-transformers faiss-cpu rank-bm25 groq pypdf python-docx python-pptx openpyxl pandas pytesseract pdf2image yt-dlp openai-whisper streamlit numpy scikit-learn
```

---

## 🧪 2. Running Automated Test Suites

The project contains automated verification suites located in the `scratch/` directory:

```bash
# 1. Run Authentication, JWT & RBAC Verification Test
python scratch/test_auth_rbac.py

# 2. Run 10-Metric Quantitative Evaluation Benchmark
python scratch/test_evaluation_dashboard.py

# 3. Run Hybrid Retrieval, CrossEncoder & RRF Test
python scratch/test_retrieval_improvements.py

# 4. Run Conversational Memory & Sliding Window Test
python scratch/test_memory_integration.py
```

### Static Analysis & Import Verification
To verify package imports and syntax without error:
```bash
python -m compileall src
python -m py_compile ui/app.py
```

---

## 🧩 3. Writing Custom Document Parsers

All document parsers inherit from `BaseParser` in `src/parsers/base.py`:

```python
from pathlib import Path
from typing import Dict, Any, List
from parsers.base import BaseParser

class CustomJsonParser(BaseParser):
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """
        Parses custom JSON files and extracts structured text.
        Returns dict with 'text' and 'metadata'.
        """
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        extracted_text = data.get("content", "")
        return {
            "text": extracted_text,
            "metadata": {
                "source": file_path.name,
                "source_type": "json",
                "char_count": len(extracted_text)
            }
        }
```

Register the new parser in `DocumentProcessor` (`src/ingestion/document_processor.py`):
```python
self.parsers = {
    ".pdf": PDFParser(),
    ".docx": DocxParser(),
    ".json": CustomJsonParser(),
    # ...
}
```

---

## 🌐 4. Adding Enterprise Connectors

All connectors are located in `src/connectors/`:
- Add a new file (e.g. `slack_connector.py` or `notion_connector.py`).
- Implement fetching & text extraction.
- Return standard chunk dictionaries:
  ```python
  {
      "text": "Extracted text content...",
      "metadata": {
          "source": "Slack Channel #general",
          "source_type": "slack",
          "user_id": user_id
      }
  }
  ```
- Pass extracted chunks to `embedding_manager.add_chunks(chunks)` and `keyword_search.add_chunks(chunks)`.
