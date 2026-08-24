import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from core.config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, CHUNKS_PATH, BM25_INDEX_PATH, GROQ_API_KEY
from core.embedding_manager import EmbeddingManager
from retrieval.keyword_search import KeywordSearch
from retrieval.retriever import HybridRetriever
from llm.generator import LLMGenerator
from core.pipeline import RAGPipeline

def run_diagnostics():
    print("=" * 80)
    print("RAG QUERY PIPELINE DIAGNOSTIC REPORT")
    print("=" * 80)

    # 1. EMBEDDINGS DIAGNOSTIC
    print("\n--- STEP 3: EMBEDDING DIAGNOSTICS ---")
    model_name = EMBEDDING_MODEL_NAME or "all-MiniLM-L6-v2"
    print(f"Configured Embedding Model: {model_name}")
    
    emb_mgr = EmbeddingManager(model_name=model_name, index_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH)
    print(f"Embedding Engine Class: {type(emb_mgr.embedding_engine).__name__}")
    print(f"Embedding Engine Model Name: {getattr(emb_mgr.embedding_engine, 'model_name', 'N/A')}")
    print(f"Embedding Dimension: {emb_mgr.dimension}")

    doc_sample = "This is a sample document passage about online regression and machine learning algorithms."
    query_sample = "What is online regression?"

    doc_emb = emb_mgr.generate_embeddings([doc_sample])
    query_emb = emb_mgr.generate_embeddings([query_sample])

    print(f"Doc Embedding Shape: {doc_emb.shape}, Norm: {np.linalg.norm(doc_emb[0]):.4f}")
    print(f"Query Embedding Shape: {query_emb.shape}, Norm: {np.linalg.norm(query_emb[0]):.4f}")
    dot_product = float(np.dot(doc_emb[0], query_emb[0]))
    print(f"Direct Cosine Similarity (Dot Product of normalized vectors): {dot_product:.4f}")

    # 2. VECTOR DATABASE DIAGNOSTIC
    print("\n--- STEP 4: VECTOR DATABASE DIAGNOSTICS ---")
    print(f"Index File Path: {FAISS_INDEX_PATH}")
    print(f"Chunks File Path: {CHUNKS_PATH}")
    print(f"FAISS Index Present: {emb_mgr.index is not None}")
    if emb_mgr.index is not None:
        print(f"FAISS Index ntotal: {emb_mgr.index.ntotal}")
        print(f"FAISS Index Dimension d: {emb_mgr.index.d}")
        print(f"FAISS Metric Type: {emb_mgr.index.metric_type} (0=IP/InnerProduct, 1=L2)")
    print(f"Loaded Chunks Count in Memory: {len(emb_mgr.chunks)}")

    # 3. RETRIEVAL & RERANKER SCORES DIAGNOSTIC
    print("\n--- STEP 5 & 6: RETRIEVAL & RERANKER DIAGNOSTICS ---")
    kw_search = KeywordSearch(index_path=BM25_INDEX_PATH)
    
    # Check if existing index has chunks, else add synthetic test chunks
    if not emb_mgr.chunks:
        print("Note: Index empty on disk, ingesting synthetic test document matching user case...")
        test_chunks = [
            {"text": f"SDA COURSE MATERIAL - Section {i}: Online regression updates linear model coefficients dynamically using online gradient descent algorithm as streaming data arrives.", "metadata": {"source": "SDACOURSEMATERIAL.docx", "page_number": i}}
            for i in range(1, 15)
        ]
        emb_mgr.add_chunks(test_chunks, save=False)
        kw_search.add_chunks(test_chunks)

    retriever = HybridRetriever(emb_mgr, kw_search)
    query = "What is online regression?"
    print(f"\nTest Query: '{query}'")

    # Run retrieval
    retrieved = retriever.retrieve(query, top_k=5)
    print(f"Number of Chunks Retrieved: {len(retrieved)}")

    for idx, c in enumerate(retrieved):
        print(f"\n--- Chunk [{idx + 1}] ---")
        meta = c.get('metadata') or {}
        text = c.get('text') or ''
        print(f"Source: {meta.get('source')}")
        print(f"Text Snippet: {text[:120]}...")
        print(f"Raw Rerank Logit Score: {c.get('raw_rerank_score')}")
        print(f"Rerank Sigmoid Probability Score: {c.get('rerank_score')}")
        print(f"FAISS Score: {c.get('score')}")

    # 4. PIPELINE CONFIDENCE & THRESHOLD DIAGNOSTIC
    print("\n--- STEP 7, 8 & 9: PIPELINE & CONTEXT DIAGNOSTICS ---")
    llm = LLMGenerator()
    pipeline = RAGPipeline(retriever, llm, confidence_threshold=0.15)
    
    print(f"Pipeline Confidence Threshold: {pipeline.confidence_threshold}")
    best_score = (retrieved[0].get('rerank_score') or 0.0) if retrieved else 0.0
    print(f"Calculated Best Score (retrieved[0]['rerank_score']): {best_score:.6f}")
    print(f"Does best_score >= confidence_threshold? {best_score >= pipeline.confidence_threshold}")

    # Test bypassing reranker
    print("\n--- TESTING RERANKER BYPASS ---")
    retrieved_no_rerank = retriever.retrieve(query, top_k=5, use_reranker=False)
    if retrieved_no_rerank:
        top_cosine = retrieved_no_rerank[0].get('score') or 0.0
        print(f"Top 1 Cosine Score (no reranker): {top_cosine:.6f}")

    # Test pipeline run
    answer, meta = pipeline.run(query)
    print("\n--- PIPELINE RUN RESULT ---")
    print(f"Answer Output:\n{answer}")
    print(f"Metadata Confidence: {meta.get('confidence')}")

if __name__ == "__main__":
    run_diagnostics()
