# 🏛️ System Architecture & Engineering Specifications

This document details the software architecture, component relationships, query execution sequence, ingestion data flow, and repository organization of the **DocuMind AI Enterprise Document Intelligence System**.

---

## 🧩 1. Component Diagram

The system follows a decoupled, modular architecture comprising five primary layers: **UI Layer**, **Security & Auth Layer**, **RAG Orchestration Layer**, **Retrieval & Embedding Layer**, and the **Persistence Storage Layer**.

```mermaid
graph TD
    User["👤 Authenticated Client"] -->|HTTPS / WebSockets| UI["🖥️ Streamlit App UI (app.py)"]
    
    subgraph Security Layer
        Auth["🔒 Auth Module (auth.py)"]
        JWT["🔑 JWT Session Manager"]
        RBAC["🛡️ RBAC Scoper (Admin / User)"]
    end
    
    UI <--> Auth
    Auth <--> JWT
    Auth <--> RBAC

    subgraph Core Pipeline Layer
        Pipeline["⚙️ RAG Pipeline (pipeline.py)"]
        Memory["🧠 Conversational Memory (chat_history.py)"]
        Generator["🤖 LLM Generator (generator.py)"]
        Evaluator["📈 RAG Evaluator (evaluator.py)"]
    end

    UI --> Pipeline
    Pipeline <--> Memory
    Pipeline --> Generator
    Evaluator --> Pipeline

    subgraph Hybrid Retrieval Layer
        Retriever["🔎 Hybrid Retriever (retriever.py)"]
        RRF["⚡ Reciprocal Rank Fusion"]
        CrossEncoder["🎯 CrossEncoder Re-ranker (ms-marco)"]
        MMR["🔀 MMR Diversifier"]
        Compressor["✂️ Context Compressor"]
    end

    Pipeline --> Retriever
    Retriever --> RRF
    RRF --> CrossEncoder
    CrossEncoder --> MMR
    MMR --> Compressor

    subgraph Storage Layer
        FAISS[("⚡ FAISS Vector Store")]
        BM25[("🔍 BM25 Lexical Store")]
        SQLite[("🛢️ SQLite DB (chat_history.db)")]
        Disk[("📂 Document Storage (/data)")]
    end

    Retriever <--> FAISS
    Retriever <--> BM25
    Memory <--> SQLite
    Auth <--> SQLite
```

---

## 🔄 2. Query Execution Sequence Diagram

The sequence diagram below traces a user query from submission to LLM answer generation, source citation, and feedback recording.

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 User Client
    participant UI as 🖥️ Streamlit UI
    participant Auth as 🔒 Auth & JWT
    participant Pipeline as ⚙️ RAG Pipeline
    participant History as 🧠 Chat History DB
    participant LLM as 🤖 LLM Generator
    participant Retriever as 🔎 Hybrid Retriever
    participant FAISS as ⚡ FAISS Index
    participant BM25 as 🔍 BM25 Index

    User->>UI: Submit Question ("What architecture does Alpha Corp design?")
    UI->>Auth: Validate JWT Session Token & Scoped User ID
    Auth-->>UI: Token Valid (User ID: u_123, Role: user)
    
    UI->>Pipeline: run(query, user_id, session_id)
    Pipeline->>History: load_session_history(user_id, session_id)
    History-->>Pipeline: Returns recent 10 dialog turns
    
    Pipeline->>LLM: rewrite_query(query, history)
    LLM-->>Pipeline: Standalone Query ("What hardware architecture does Alpha Corp design?")
    
    Pipeline->>Retriever: retrieve(standalone_query, top_k=5, user_id=u_123)
    
    par Vector Search
        Retriever->>FAISS: Vector Similarity Search (FAISS)
        FAISS-->>Retriever: Top-K Vector Chunks
    and Keyword Search
        Retriever->>BM25: BM25 Lexical Keyword Search
        BM25-->>Retriever: Top-K Keyword Chunks
    end
    
    Retriever->>Retriever: Apply Reciprocal Rank Fusion (RRF)
    Retriever->>Retriever: CrossEncoder Re-ranking (ms-marco-MiniLM-L-6-v2)
    Retriever->>Retriever: MMR Diversification & Context Compression
    Retriever-->>Pipeline: Return Top Filtered & Re-ranked Chunks
    
    Pipeline->>LLM: generate_answer(standalone_query, chunks, history)
    LLM-->>Pipeline: Generated Answer Stream + Source Citations
    
    Pipeline->>History: save_chat(user_id, role, content, confidence, sources)
    Pipeline-->>UI: Render Streaming Answer, Badges, Citations & Feedback Buttons
    
    User->>UI: Click 👍 Thumbs Up Feedback
    UI->>History: save_feedback(user_id, query, "thumbs_up")
```

---

## 🌊 3. Ingestion & Data Flow Diagram

Demonstrates how documents from 13 enterprise sources are parsed, chunked, embedded, and stored across FAISS and BM25 indexes.

```mermaid
flowchart LR
    subgraph Multi-Source Ingestion
        PDF["📄 PDF / OCR"]
        Office["📊 Word / PPT / Excel"]
        Media["▶️ YouTube / Video"]
        Web["🌐 Web / GitHub / DBs"]
        GDrive["📥 Google Drive"]
    end

    subgraph Document Processing
        Processor["⚙️ Document Processor (document_processor.py)"]
        Parsers["🧩 Specialized Parsers (PDF, DOCX, CSV, Image OCR)"]
        Chunker["✂️ Sentence-Aware Overlapping Chunker"]
    end

    subgraph Dual Storage Indexing
        EmbManager["⚡ Embedding Manager (embedding_manager.py)"]
        FAISS_Idx["⚡ FAISS Vector Index (.bin)"]
        BM25_Idx["🔍 BM25 Lexical Index (.pkl)"]
        MetaDB["🛢️ SQLite Document Metadata"]
    end

    PDF --> Processor
    Office --> Processor
    Media --> Processor
    Web --> Processor
    GDrive --> Processor

    Processor --> Parsers
    Parsers --> Chunker
    Chunker --> EmbManager

    EmbManager --> FAISS_Idx
    EmbManager --> BM25_Idx
    EmbManager --> MetaDB
```

---

## 📂 4. Repository Directory Structure

```
Ai_Document_Intelligence_System/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD Pipeline
├── docs/                          # Comprehensive Documentation Suite
│   ├── ARCHITECTURE.md            # System Architecture & Mermaid Diagrams
│   ├── API_DOCUMENTATION.md       # Python API Specifications
│   ├── DEVELOPER_GUIDE.md         # Developer Setup & Testing Guide
│   ├── USER_GUIDE.md              # End-User UI Manual
│   ├── TROUBLESHOOTING.md         # Diagnostic & Fix Handbook
│   └── FUTURE_SCOPE.md            # Product Roadmap & Architectural Next Steps
├── nginx/
│   └── nginx.conf                 # Nginx SSL Reverse Proxy & Rate Limiting
├── src/                           # Primary Application Source Code
│   ├── connectors/                # Enterprise Source Connectors
│   │   ├── drive_connector.py     # Google Drive API Integrator
│   │   ├── github_connector.py    # GitHub Repository Parser
│   │   ├── table_connector.py     # SQL Database & Structured Table Connector
│   │   ├── web_crawler.py         # Web Crawling & Scraping Connector
│   │   └── youtube_connector.py   # YouTube Transcript & Video Audio Parser
│   ├── core/                      # Engine Infrastructure
│   │   ├── auth.py                # User Authentication, Password Salting & JWT
│   │   ├── chat_history.py        # SQLite Persistence, Session Memory & Feedback
│   │   ├── config.py              # System Configuration & Path Constants
│   │   ├── embedding_manager.py   # PyTorch SentenceTransformers & FAISS Engine
│   │   ├── logger.py              # Structured System Logging Infrastructure
│   │   └── pipeline.py            # RAG Pipeline Orchestrator with Latency Metrics
│   ├── evaluation/
│   │   └── evaluator.py           # 10-Metric Quantitative Benchmark Evaluator
│   ├── ingestion/
│   │   └── document_processor.py  # Master Ingestion Router
│   ├── llm/
│   │   ├── base.py                # Abstract LLM Interface
│   │   ├── generator.py           # LLM Response Generator & Standalone Rewriter
│   │   └── groq_client.py         # Groq Llama-3 API Engine Client
│   ├── ocr/
│   │   └── tesseract_ocr.py       # Image OCR Text Extraction Engine
│   ├── parsers/                   # Document Format Parsers
│   │   ├── base.py                # Base Parser Contract
│   │   ├── docx_parser.py         # MS Word (.docx) Parser
│   │   ├── html_parser.py         # HTML & Web Document Parser
│   │   ├── markdown_parser.py     # Markdown (.md) Parser
│   │   ├── pdf_parser.py          # PDF Parser (pypdf + pdf2image fallback)
│   │   ├── pptx_parser.py         # MS PowerPoint (.pptx) Parser
│   │   ├── spreadsheet_parser.py  # Excel & CSV (.xlsx, .csv) Parser
│   │   └── text_parser.py         # Plain Text (.txt) Parser
│   └── retrieval/
│       ├── keyword_search.py      # BM25 Lexical Keyword Engine
│       └── retriever.py           # Upgraded Hybrid Retriever (RRF, CrossEncoder, MMR)
├── ui/
│   └── app.py                     # Streamlit Enterprise Interface
├── .dockerignore                  # Excluded Docker Build Files
├── .env.example                   # Environment Configuration Template
├── DEPLOYMENT.md                  # Docker & NVIDIA GPU Setup Guide
├── docker-compose.yml             # Local Docker Compose Orchestration
├── docker-compose.prod.yml        # Production Nginx + Certbot Docker Compose
├── Dockerfile                     # Multi-stage Container Buildfile
├── PRODUCTION_DEPLOYMENT.md       # Multi-Cloud & HTTPS Deployment Handbook
├── railway.json                   # Railway PaaS Deployment Manifest
├── render.yaml                    # Render PaaS Deployment Blueprint
└── pyproject.toml                 # Python Package Metadata & Dependencies
```
