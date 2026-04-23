# 📚 Advanced AI Document Intelligence System

A production-ready Retrieval-Augmented Generation (RAG) system designed to ingest, analyze, and intelligently answer questions across multiple documents. This system utilizes a highly optimized hybrid search architecture to eliminate hallucinations and provide mathematically backed, verifiable answers with exact source citations.

---

## 🌟 What is this project?
The Advanced AI Document Intelligence System is a sophisticated chat-based application that allows users to upload multiple documents (PDFs, TXTs) and query them collectively. Instead of relying on an AI's pre-trained (and potentially outdated or incorrect) knowledge, this system strictly limits the AI to only use the exact text found within your uploaded documents. 

If the answer isn't in your files, the AI will refuse to guess, guaranteeing high-fidelity intelligence for legal, medical, academic, or corporate document analysis.

---

## 🛠️ Technologies Used

### Core Architecture
*   **Python**: The core backend language.
*   **Streamlit**: Powers the interactive, chat-based frontend UI.

### Ingestion & Retrieval (RAG Pipeline)
*   **PyPDF2**: Extracts raw text from uploaded PDF files while tracking exact page numbers.
*   **Sentence-Transformers (`all-mpnet-base-v2`)**: Transforms text chunks into high-dimensional mathematical vectors for semantic understanding.
*   **FAISS (Facebook AI Similarity Search)**: An ultra-fast, memory-mapped vector database used to instantly find contextually similar text (Semantic Search).
*   **BM25 (`rank_bm25`)**: A highly-efficient lexical search engine used to find exact keyword matches (Keyword Search).

### Generation (LLM)
*   **Google Gemini API (`gemini-pro`)**: The generative engine that processes the retrieved contexts and formulates a human-readable answer.

---

## ⚙️ How it Works (The Pipeline)

1.  **Smart Ingestion**: 
    When you upload documents, the system doesn't just read them; it breaks them down into "chunks" (approx. 400-800 words) while explicitly preventing sentences or paragraphs from being cut in half. It securely tags every single chunk with its source filename and page number.
2.  **Dual Indexing**: 
    The chunks are embedded into vectors and stored in **FAISS**, while simultaneously being indexed into **BM25**.
3.  **Hybrid Search**: 
    When you ask a question, the system queries FAISS (to understand the *meaning* of your question) and BM25 (to find exact *keywords*). 
4.  **Reciprocal Rank Fusion (RRF) & Re-Ranking**: 
    The results from both search engines are merged using the mathematical RRF algorithm, and then strictly re-ranked using **Cosine Similarity** to find the absolute top 5 most relevant paragraphs.
5.  **Strict Prompting & Generation**: 
    The top contexts are injected into a highly-restrictive prompt and sent to Gemini. Gemini synthesizes the answer and explicitly cites the source documents (e.g., `[Document 1]`).
6.  **Explainability**: 
    The UI displays the final answer alongside a "Confidence Score" and an expandable section revealing the exact document names, page numbers, and quote snippets used to generate the answer.

---

## 💡 What is its Use Case?

This system is built for environments where **accuracy is non-negotiable**.
*   **Legal & Compliance**: Rapidly query hundreds of pages of contracts to find specific clauses without fear of the AI hallucinating legal terms.
*   **Academic Research**: Upload multiple research papers and ask the system to synthesize methodologies, providing exact page numbers for your bibliography.
*   **Corporate Knowledge Base**: Ingest company handbooks, HR policies, or technical manuals to provide employees with instant, 100% accurate internal support.

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your machine.

### 2. Install Dependencies
Navigate to the project root and install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY="your_api_key_here"
```

### 4. Run the Application
Launch the Streamlit interface:
```bash
streamlit run ui/app.py
```
*Note: The first time you run the application, it will take a few minutes to automatically download the 420MB `all-mpnet-base-v2` embedding model.*

---

## 📁 Project Structure

```text
├── data/                  # Raw uploaded documents
├── embeddings/            # Persistent FAISS and BM25 indices
├── logs/                  # System logs and evaluation reports
├── src/
│   ├── core/              # Config, Logging, Pipeline Orchestration, & Embedding caching
│   ├── ingestion/         # PDF Parsing and Smart Chunking
│   ├── retrieval/         # BM25 implementation & Hybrid Retriever
│   ├── llm/               # Gemini API wrapper & Prompt Engineering
│   └── evaluation/        # Automated accuracy benchmarking scripts
├── ui/
│   └── app.py             # Streamlit frontend
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
```
