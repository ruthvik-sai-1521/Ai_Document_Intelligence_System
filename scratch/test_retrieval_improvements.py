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

TEST_DIR = PROJECT_ROOT / "scratch" / "test_retrieval_indices"
TEST_FAISS = TEST_DIR / "faiss.bin"
TEST_CHUNKS = TEST_DIR / "chunks.pkl"
TEST_BM25 = TEST_DIR / "bm25.pkl"

def setup_test_index():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Managers
    emb_mgr = EmbeddingManager(index_path=TEST_FAISS, chunks_path=TEST_CHUNKS)
    kw_search = KeywordSearch(index_path=TEST_BM25)
    
    # 2. Add sample chunks with diverse semantic content and metadata properties
    sample_chunks = [
        {
            "text": "The Alpha Corporation is a hardware manufacture enterprise. They design next-generation quantum GPUs for artificial intelligence architectures.",
            "metadata": {"source": "alpha.txt", "source_type": "text", "user_id": "user1"}
        },
        {
            "text": "The Alpha Corporation is a hardware manufacture enterprise. They design next-generation quantum GPUs for artificial intelligence architectures.",
            "metadata": {"source": "alpha_duplicate.txt", "source_type": "text", "user_id": "user1"}
        },
        {
            "text": "Gold reserves have increased by 15 percent over the last financial quarter. Inflation rates remain flat globally.",
            "metadata": {"source": "finance.txt", "source_type": "text", "user_id": "user1"}
        },
        {
            "text": "The Beta Corporation develops software platforms. Their main products are developer tools and integrated development environments.",
            "metadata": {"source": "beta.txt", "source_type": "text", "user_id": "user2"}
        }
    ]
    
    emb_mgr.add_chunks(sample_chunks)
    kw_search.add_chunks(sample_chunks)
    
    return emb_mgr, kw_search

def test_upgrades():
    print("=" * 80)
    print("STARTING RETRIEVAL IMPROVEMENTS VERIFICATION")
    print("=" * 80)
    
    emb_mgr, kw_search = setup_test_index()
    retriever = HybridRetriever(emb_mgr, kw_search)
    
    # Check model loading
    print("→ Testing lazy-loaded Cross-Encoder model...")
    assert retriever.cross_encoder is not None
    print("✓ Cross-Encoder model loaded successfully\n")
    
    # 1. Test Metadata Filtering (Filter by user_id = user1 vs user2)
    print("→ Testing Metadata Filtering (User scoping)...")
    res_user1 = retriever.retrieve("Alpha Corporation", top_k=5, filters={"user_id": "user1"})
    res_user2 = retriever.retrieve("Alpha Corporation", top_k=5, filters={"user_id": "user2"})
    
    assert len(res_user1) > 0
    assert all(c["metadata"]["user_id"] == "user1" for c in res_user1)
    print(f"✓ Metadata filter user1 matched {len(res_user1)} chunks.")
    
    assert len(res_user2) > 0
    assert all(c["metadata"]["user_id"] == "user2" for c in res_user2)
    print(f"✓ Metadata filter user2 matched {len(res_user2)} chunks.")
    print("✓ Metadata Filtering Passed\n")
    
    # 2. Test Cross-Encoder Re-ranking
    print("→ Testing Cross-Encoder Re-ranking...")
    # Retrieve without filter
    res_rerank = retriever.retrieve("Who designs quantum GPUs?", top_k=2, use_reranker=True)
    assert len(res_rerank) > 0
    # The GPU chunk should be ranked first
    assert "quantum GPUs" in res_rerank[0]["text"]
    assert "rerank_score" in res_rerank[0]
    print(f"✓ Top chunk: '{res_rerank[0]['text'][:40]}...' Score: {res_rerank[0]['rerank_score']}")
    print("✓ Cross-Encoder Re-ranking Passed\n")
    
    # 3. Test MMR (Maximal Marginal Relevance) Diversity Selection
    print("→ Testing MMR diversity selection...")
    # Query for Alpha Corp. The index has two identical chunks (alpha.txt and alpha_duplicate.txt).
    # With MMR (lambda=0.5), it should select one alpha chunk and prioritize the next diverse match (finance.txt) over the second duplicate chunk.
    res_mmr = retriever.retrieve(
        "Alpha Corporation and quantum GPUs",
        top_k=2,
        use_mmr=True,
        use_compression=False,
        use_adaptive_k=False,
        lambda_val=0.5
    )
    
    assert len(res_mmr) == 2
    sources = [c["metadata"]["source"] for c in res_mmr]
    # Check that both selected chunks are not duplicates
    assert "alpha.txt" in sources or "alpha_duplicate.txt" in sources
    assert "finance.txt" in sources
    print(f"✓ MMR selected diverse sources: {sources}")
    print("✓ MMR Diversity Selection Passed\n")
    
    # 4. Test Context Compression (sentence extraction)
    print("→ Testing Context Compression...")
    # The chunk finance.txt has two sentences: one about gold reserves, one about inflation.
    # A query specifically about "gold reserves" should compress the chunk, keeping only the gold reserves sentence.
    res_comp = retriever.retrieve(
        "What happened to gold reserves?",
        top_k=1,
        use_mmr=False,
        use_compression=True,
        use_adaptive_k=False,
        compression_threshold=0.35
    )
    
    assert len(res_comp) == 1
    comp_chunk = res_comp[0]
    assert comp_chunk.get("compressed") is True
    assert "Gold reserves" in comp_chunk["text"]
    assert "Inflation rates" not in comp_chunk["text"]
    print("✓ Original text:", comp_chunk["original_text"])
    print("✓ Compressed text:", comp_chunk["text"])
    print("✓ Context Compression Passed\n")
    
    # 5. Test Adaptive Top-K Pruning
    print("→ Testing Adaptive Top-K score drop pruning...")
    # For a query with a single highly relevant match, other low-relevance chunks should be pruned.
    res_adaptive = retriever.retrieve(
        "Who designs quantum GPUs?",
        top_k=4,
        use_reranker=True,
        use_mmr=False,
        use_compression=False,
        use_adaptive_k=True
    )
    # The top chunk is extremely relevant. The finance chunk should be pruned because it doesn't mention GPUs at all.
    assert len(res_adaptive) < 4
    print(f"✓ Adaptive Top-K retrieved {len(res_adaptive)} chunks instead of maximum 4.")
    print("✓ Adaptive Top-K Passed\n")
    
    # Cleanup
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        
    print("=" * 80)
    print("ALL RETRIEVAL IMPROVEMENTS TEST CASES VERIFIED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_upgrades()
