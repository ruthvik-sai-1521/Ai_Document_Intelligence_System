import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from core.config import BM25_INDEX_PATH
from core.embedding_manager import EmbeddingManager
from retrieval.keyword_search import KeywordSearch
from retrieval.retriever import HybridRetriever
from llm.generator import LLMGenerator
from core.pipeline import RAGPipeline

def test_pipeline():
    print("=" * 80)
    print("TESTING RAG PIPELINE SCORE DISTRIBUTION AND REJECTION")
    print("=" * 80)

    emb_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    kw_search = KeywordSearch(index_path=BM25_INDEX_PATH)

    # Simulate realistic document chunks from a course material file
    sample_chunks = [
        {"text": "SDA COURSE MATERIAL - Chapter 1: Machine Learning Foundations. Online regression is a supervised learning method where the model updates its parameters incrementally as each data point arrives in real-time stream.", "metadata": {"source": "SDACOURSEMATERIAL.docx", "page_number": 1}},
        {"text": "SDA COURSE MATERIAL - Chapter 2: Gradient Descent and Optimization. Stochastic gradient descent is commonly used in online regression to compute parameter updates efficiently with O(1) memory complexity.", "metadata": {"source": "SDACOURSEMATERIAL.docx", "page_number": 2}},
        {"text": "SDA COURSE MATERIAL - Chapter 3: Evaluation Metrics. Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) measure performance in online regression tasks.", "metadata": {"source": "SDACOURSEMATERIAL.docx", "page_number": 3}},
    ]

    emb_mgr.add_chunks(sample_chunks, save=False)
    kw_search.add_chunks(sample_chunks)

    retriever = HybridRetriever(emb_mgr, kw_search)
    llm = LLMGenerator()

    query = "Explain online regression and how gradient descent is used"
    print(f"\nUser Query: '{query}'")

    print("\n--- STEP 1: HYBRID RETRIEVAL & RERANKER ---")
    retrieved = retriever.retrieve(query, top_k=3)

    for i, c in enumerate(retrieved):
        print(f"\nChunk [{i+1}]:")
        print(f"  Snippet: {c['text'][:100]}...")
        print(f"  Raw Rerank Logit (MS-MARCO): {c.get('raw_rerank_score'):.4f}")
        print(f"  Sigmoid Score: {c.get('rerank_score'):.6f}")

    top_sig_score = retrieved[0].get('rerank_score', 0.0) if retrieved else 0.0

    print("\n--- STEP 2: PIPELINE EVALUATION AT DIFFERENT THRESHOLDS ---")
    for thresh in [0.15, 0.05, 0.01, 0.001, 0.0001, 0.0]:
        pipe = RAGPipeline(retriever, llm, confidence_threshold=thresh)
        ans, meta = pipe.run(query)
        is_rejected = ans == "Insufficient data"
        print(f"Threshold: {thresh:<6} | Best Score: {top_sig_score:.6f} | Rejected: {is_rejected} | Answer Preview: {ans[:60]}")

if __name__ == "__main__":
    test_pipeline()
