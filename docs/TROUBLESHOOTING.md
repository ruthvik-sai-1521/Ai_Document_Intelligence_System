# 🔍 Troubleshooting & Diagnostic Guide

This guide provides solutions for common runtime, build, static analysis, and configuration issues in the **DocuMind AI Document Intelligence System**.

---

## 🛑 1. Common Diagnostics & Solutions

### Issue 1: `Groq API Key Missing or Invalid`
- **Symptom**: Error log shows `groq.AuthenticationError` or LLM generator returns `"An error occurred during response generation"`.
- **Cause**: `GROQ_API_KEY` is not set or invalid in `.env`.
- **Solution**:
  1. Obtain a valid Groq API key from [https://console.groq.com](https://console.groq.com).
  2. Add key to `.env`: `GROQ_API_KEY=gsk_your_key_here`.
  3. Restart Streamlit or Docker container (`docker compose restart`).

---

### Issue 2: `Static Analyzer (Pyright/Pylance) Cannot Find Module`
- **Symptom**: VS Code displays red squiggly lines under `from core.config import ...` or `from retrieval.retriever import ...`.
- **Cause**: Pyright import root is configured to `src/` layout.
- **Solution**:
  Verify `pyrightconfig.json` exists in project root:
  ```json
  {
      "include": ["src", "ui"],
      "extraPaths": ["src"],
      "executionEnvironments": [
          { "root": "src" }
      ]
  }
  ```
  Ensure all subdirectories in `src/` contain `__init__.py` files (`src/core/__init__.py`, `src/retrieval/__init__.py`, `src/llm/__init__.py`, etc.).

---

### Issue 3: `CUDA Out of Memory (OOM)`
- **Symptom**: PyTorch throws `torch.cuda.OutOfMemoryError` during CrossEncoder re-ranking or embedding generation.
- **Cause**: Batch size or model loading exceeding GPU VRAM.
- **Solution**:
  - The `HybridRetriever` uses lazy loading for the CrossEncoder model.
  - Set `CUDA_VISIBLE_DEVICES=""` in `.env` to fallback gracefully to CPU execution if GPU VRAM is restricted.

---

### Issue 4: `sqlite3.OperationalError: database is locked`
- **Symptom**: SQLite error during multi-user write operations.
- **Cause**: Simultaneous database writes across multiple threads.
- **Solution**:
  `chat_history.py` uses context manager connections (`with _get_connection() as conn:`). Verify timeout is set:
  ```python
  conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
  ```

---

### Issue 5: `TesseractNotFoundError` during Image OCR
- **Symptom**: Error when uploading `.png` or `.jpg` files: `tesseract is not installed or it's not in your PATH`.
- **Cause**: Tesseract OCR binary missing on host system.
- **Solution**:
  - **Linux**: `sudo apt-get install -y tesseract-ocr tesseract-ocr-eng`
  - **Windows**: Download Tesseract installer from GitHub and add `C:\Program Files\Tesseract-OCR` to System `PATH`.
  - **Docker**: Pre-installed in `Dockerfile`.

---

### Issue 6: Streamlit Port 8501 Conflict
- **Symptom**: `OSError: [Errno 98] Address already in use`.
- **Solution**:
  Kill active Streamlit processes or change port:
  ```bash
  streamlit run ui/app.py --server.port=8502
  ```
