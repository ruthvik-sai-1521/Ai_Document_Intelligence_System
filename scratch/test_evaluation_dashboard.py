import sys
from pathlib import Path
import os
import shutil

# Add project root and src folder to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Configure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.core.embedding_manager import EmbeddingManager
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.retriever import HybridRetriever
from src.llm.generator import LLMGenerator
from src.core.pipeline import RAGPipeline
from src.evaluation.evaluator import RAGEvaluator

TEST_DIR = PROJECT_ROOT / "scratch" / "test_eval_indices"
TEST_FAISS = TEST_DIR / "faiss.bin"
TEST_CHUNKS = TEST_DIR / "chunks.pkl"
TEST_BM25 = TEST_DIR / "bm25.pkl"

def setup_test_index():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    emb_mgr = EmbeddingManager(index_path=TEST_FAISS, chunks_path=TEST_CHUNKS)
    kw_search = KeywordSearch(index_path=TEST_BM25)
    
    sample_chunks = [
        {
            "text": "The Alpha Corporation is a hardware manufacture enterprise. They design next-generation quantum GPUs for artificial intelligence architectures.",
            "metadata": {"source": "alpha.txt", "source_type": "text", "user_id": "test_user"}
        }
    ]
    
    emb_mgr.add_chunks(sample_chunks)
    kw_search.add_chunks(sample_chunks)
    
    return emb_mgr, kw_search

def test_evaluation():
    print("=" * 80)
    print("STARTING EVALUATION DASHBOARD METRICS VERIFICATION")
    print("=" * 80)
    
    emb_mgr, kw_search = setup_test_index()
    llm = LLMGenerator()
    retriever = HybridRetriever(emb_mgr, kw_search)
    pipeline = RAGPipeline(retriever, llm)
    
    evaluator = RAGEvaluator(pipeline, results_dir=str(TEST_DIR))
    
    # 1. Single query evaluation
    print("→ Running single query evaluation...")
    query = "What does Alpha Corporation design?"
    metrics = evaluator.evaluate_single_query(query, expected_keywords=["quantum", "gpu", "hardware"])
    
    print("\n--- SINGLE QUERY METRIC RESULTS ---")
    required_metrics = [
        "retrieval_precision", "recall", "latency_seconds", "embedding_time_ms",
        "llm_response_time", "faithfulness", "context_relevance", "answer_relevance",
        "citation_accuracy", "hallucination_rate"
    ]
    
    for metric_name in required_metrics:
        assert metric_name in metrics, f"Missing required metric: {metric_name}"
        print(f"  • {metric_name.replace('_', ' ').title()}: {metrics[metric_name]}")
        
    print("✓ Single query evaluation metrics validated successfully!\n")
    
    # 2. System Benchmark Run
    print("→ Running system benchmark suite...")
    summary = evaluator.run_benchmark()
    
    print("\n--- BENCHMARK SUMMARY METRICS ---")
    for k, v in summary.items():
        if k != "details":
            print(f"  • {k.replace('_', ' ').title()}: {v}")
            
    assert "retrieval_precision" in summary
    assert "recall" in summary
    assert "latency" in summary
    assert "embedding_time_ms" in summary
    assert "llm_response_time" in summary
    assert "faithfulness" in summary
    assert "context_relevance" in summary
    assert "answer_relevance" in summary
    assert "citation_accuracy" in summary
    assert "hallucination_rate" in summary
    
    print("\n✓ System Benchmark Suite executed and verified successfully!")
    
    # Cleanup
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        
    print("=" * 80)
    print("ALL 10 EVALUATION METRICS VERIFIED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_evaluation()
