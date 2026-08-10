# 👤 End-User Manual & UI Interface Guide

Welcome to the **DocuMind AI Document Intelligence Platform** user guide. This document explains how to navigate the Streamlit interface, manage documents, query intelligence sources, review citations, rate responses, and inspect system evaluation benchmarks.

---

## 🔑 1. Authentication & Role Selection

When opening the platform, you will be greeted by the **🔒 Enterprise Authentication & RBAC Portal**:

1. **Sign In**:
   - Enter your `Username` and `Password`.
   - **Admin Account**: Username `admin` | Password `admin123` (Full system access across all users).
   - **User Account**: Username `user` | Password `user123` (Scoped document privacy).
2. **Register Account**:
   - Create a new account with your preferred username, password, and role.

---

## 📂 2. Uploading & Ingesting Intelligence Sources

Use the **Sidebar** to ingest documents and data streams:

1. **📂 Files**: Drag and drop documents (PDF, TXT, DOCX, PPTX, XLSX, CSV, MD, HTML, PNG, JPG). Click **⚡ Process Files**.
2. **🌐 Web**: Enter any public website URL to scrape and index page text.
3. **🐙 GitHub**: Enter a public repository URL to parse code and documentation.
4. **▶️ YouTube**: Enter a YouTube video URL to automatically extract transcripts and timestamped segments.
5. **📥 Google Drive**: Paste a shared Google Drive File ID to import directly from the cloud.
6. **🛢️ Databases**: Enter SQLite, MySQL, or PostgreSQL connection URIs to query structured tables.

---

## 💬 3. Interactive Chat & Conversational Memory

1. **Asking Questions**:
   - Type your question in the bottom chat bar in the **💬 Chat** tab.
   - You can ask follow-up questions naturally (e.g. *"What did they say about inflation?"* followed by *"How does that compare to last year?"*). The system uses LLM Contextual Query Rewriting to preserve context.
2. **Confidence Badges & Reference Citations**:
   - Each answer displays a confidence badge:
     - `● High Confidence`: Score $\ge 0.50$
     - `● Med Confidence`: Score $\ge 0.25$
     - `● Low Confidence`: Score $< 0.25$
   - Expand **🔗 Reference Citations** to view exact document sources, page numbers, video timestamps, and re-ranking quality scores.
3. **Interactive Feedback**:
   - Rate any answer using the **👍 Thumbs Up** or **👎 Thumbs Down** buttons beneath the assistant response. Your feedback is stored in the database to improve model performance.

---

## 👁️ 4. Document Explorer & Chunk Inspector

Open the **📂 Documents** tab to manage your indexed document library:

1. **Quick Keyword Search**: Type any term to find direct snippet matches across all stored text.
2. **👁️ Chunk Inspector**: Click the eye icon **👁️** next to any listed document to open the Document Previewer drawer. This displays all parsed text chunks, word counts, and page numbers for that document.
3. **🗑️ Delete**: Click the trash icon **🗑️** to completely remove a document from FAISS, BM25, and SQLite.

---

## 📈 5. System Evaluation Dashboard

Open the **📈 Evaluation Dashboard** tab to view quantitative system quality and latency metrics:

- **10 Core Metrics**:
  1. **Retrieval Precision**: Ratio of relevant retrieved chunks.
  2. **Recall**: Coverage of ground-truth concepts.
  3. **Faithfulness**: Sentence-level grounding ratio in source documents.
  4. **Context Relevance**: Semantic alignment of retrieved context.
  5. **Answer Relevance**: Semantic alignment between query and response.
  6. **Citation Accuracy**: Validity of source citations.
  7. **Hallucination Rate**: $1.0 - \text{Faithfulness}$ (rate of ungrounded statements).
  8. **Total Latency**: Wall-clock response time (seconds).
  9. **Embedding Time**: High-precision vector calculation time (milliseconds).
  10. **LLM Response Time**: Time spent generating tokens via LLM API (seconds).
- **🚀 Run System Evaluation Benchmark**: Click the button to execute a live 4-query ground-truth benchmark suite across the active index!
