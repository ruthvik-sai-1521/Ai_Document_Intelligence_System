---
title: DocuMind AI Enterprise RAG
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.30.0
app_file: ui/app.py
pinned: false
---

<div align="center">

# 🧠 DocuMind AI — Enterprise Document Intelligence System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch GPU](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C.svg)](https://pytorch.org/)
[![Docker Supported](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*An enterprise-grade, multi-source RAG system featuring Hybrid Semantic+Lexical Search, CrossEncoder Re-ranking, Conversational Memory, JWT Authentication, RBAC Scoping, and a 10-Metric Quantitative Evaluation Dashboard.*

</div>

---

## 🌟 Key Features

* **⚡ Upgraded Hybrid Retrieval Pipeline**:
  - **Vector Search**: Dense FAISS vector store ($d=384$ normalized embeddings via `all-MiniLM-L6-v2`).
  - **Lexical Search**: Okapi BM25 keyword matching engine.
  - **Reciprocal Rank Fusion (RRF)**: Merges rank positions dynamically using $RRF\_Score = \sum \frac{1}{60 + r}$.
  - **CrossEncoder Re-ranking**: Precision candidate evaluation using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - **MMR Diversification**: Maximal Marginal Relevance ($\lambda = 0.7$) to eliminate passage redundancy.
  - **Context Compression & Adaptive Top-K**: Sentence-level semantic filtering and sharp score drop pruning.

* **📥 Multi-Source Ingestion Engine (13 Connectors)**:
  - Supports PDFs, TXT, DOCX, PPTX, XLSX, CSV, Markdown, HTML, Image OCR (Tesseract), YouTube Transcripts, GitHub Repos, SQL Databases (SQLite/MySQL/PostgreSQL), and Google Drive.

* **🧠 Conversational Memory & Token Optimization**:
  - Isolated session state persistence (`session_id`).
  - Contextual Query Rewriting for complex follow-up questions.
  - Sliding window token optimization (pruning older turns over 600 words).

* **🔒 Enterprise Security & RBAC Scoping**:
  - SHA-256 password salting and JWT session token verification.
  - Role-Based Access Control (`admin` and `user` roles).
  - Strict user-level document isolation for complete data privacy.

* **📈 10-Metric Quantitative Evaluation Dashboard**:
  - Tracks *Retrieval Precision, Recall, Latency, Embedding Time, LLM Response Time, Faithfulness, Context Relevance, Answer Relevance, Citation Accuracy, and Hallucination Rate*.
  - 1-click system benchmark execution built into Streamlit.

* **👁️ Interactive Document Previewer & Feedback**:
  - **Chunk Inspector**: Preview raw parsed text chunks, word counts, and page numbers.
  - **Interactive Feedback**: Rate assistant responses with 👍 / 👎 buttons.

---

## 🏛️ System Architecture

```
                      +-----------------------------+
                      |  Streamlit Enterprise UI    |
                      +--------------+--------------+
                                     |
                      +--------------+--------------+
                      | 🔒 JWT Auth & RBAC Security |
                      +--------------+--------------+
                                     |
                      +--------------+--------------+
                      | ⚙️ RAG Pipeline & Memory   |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+-------------------------------+               +-------------------------------+
| ⚡ FAISS Vector Search Engine |               | 🔍 Okapi BM25 Keyword Search  |
+---------------+---------------+               +---------------+---------------+
                |                                               |
                +-----------------------+-----------------------+
                                        |
                                        v
                        +---------------+---------------+
                        | ⚡ Reciprocal Rank Fusion    |
                        +---------------+---------------+
                                        |
                                        v
                        +---------------+---------------+
                        | 🎯 CrossEncoder Re-ranker     |
                        +---------------+---------------+
                                        |
                                        v
                        +---------------+---------------+
                        | 🤖 LLM Answer Generator       |
                        +-------------------------------+
```

---

## ⚡ Quickstart Guide

### 1. Local Python Setup
```bash
# Clone repository
git clone https://github.com/your-org/Ai_Document_Intelligence_System.git
cd Ai_Document_Intelligence_System

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install PyJWT sentence-transformers faiss-cpu rank-bm25 groq pypdf python-docx python-pptx openpyxl pandas pytesseract pdf2image yt-dlp openai-whisper streamlit numpy scikit-learn

# Set your Groq API Key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_your_groq_api_key_here

# Launch Streamlit Application
streamlit run ui/app.py
```

### 2. Docker Compose Deployment
```bash
cp .env.example .env
# Set GROQ_API_KEY in .env

docker compose up -d --build
# Open http://localhost:8501 in your browser
```

---

## 📚 Complete Documentation Index

| Documentation Guide | Description |
| :--- | :--- |
| 🏛️ **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Technical Architecture, Mermaid Component, Sequence & Flow Diagrams, Directory Tree |
| 📖 **[docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)** | Core Python API Specifications & Signature Specifications |
| 🛠️ **[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** | Developer Environment Setup, Running Tests & Extending Connectors |
| 👤 **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** | End-User UI Manual, Document Previewer, Chat Citations & Benchmark Dashboard |
| 🔍 **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | Diagnostic & Resolution Steps for Common Gotchas & Errors |
| 🚀 **[docs/FUTURE_SCOPE.md](docs/FUTURE_SCOPE.md)** | Product Roadmap, Agentic Tools & Architectural Next Steps |
| 🐳 **[DEPLOYMENT.md](DEPLOYMENT.md)** | Local Docker & NVIDIA GPU Passthrough Setup |
| 🚀 **[PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)** | Multi-Cloud Deployment Guide (AWS, GCP, Azure, Render, Railway, Nginx SSL) |

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
