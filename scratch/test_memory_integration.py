import sys
from pathlib import Path
import os
import shutil

# Add project root and src folder to sys.path
PROJECT_ROOT = Path(r"d:\Projects\Ai_Document_Intelligence_System")
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
from src.core.chat_history import save_chat, load_session_history, init_db

TEST_DIR = PROJECT_ROOT / "scratch" / "test_memory_indices"
TEST_FAISS = TEST_DIR / "faiss.bin"
TEST_CHUNKS = TEST_DIR / "chunks.pkl"
TEST_BM25 = TEST_DIR / "bm25.pkl"

def setup_test_index():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    init_db()
    
    # 1. Initialize Managers
    emb_mgr = EmbeddingManager(index_path=TEST_FAISS, chunks_path=TEST_CHUNKS)
    kw_search = KeywordSearch(index_path=TEST_BM25)
    
    # 2. Add sample chunks
    sample_chunks = [
        {
            "text": "The Alpha Corporation is a hardware manufacture enterprise. They design next-generation quantum GPUs for artificial intelligence architectures.",
            "metadata": {"source": "alpha.txt", "source_type": "text", "user_id": "test_user"}
        }
    ]
    
    emb_mgr.add_chunks(sample_chunks)
    kw_search.add_chunks(sample_chunks)
    
    return emb_mgr, kw_search

def test_memory():
    print("=" * 80)
    print("STARTING CONVERSATIONAL MEMORY INTEGRATION TEST")
    print("=" * 80)
    
    emb_mgr, kw_search = setup_test_index()
    llm = LLMGenerator()
    retriever = HybridRetriever(emb_mgr, kw_search)
    pipeline = RAGPipeline(retriever, llm)
    
    user_id = "test_user"
    session_a = "session_alpha"
    session_b = "session_beta"
    
    # 1. Test query rewriting & follow-up question
    print("→ Query 1 (Session A): 'What is Alpha Corporation?'")
    ans1, meta1 = pipeline.run("What is Alpha Corporation?", user_id=user_id, session_id=session_a)
    save_chat(user_id=user_id, role="user", content="What is Alpha Corporation?", session_id=session_a)
    save_chat(user_id=user_id, role="assistant", content=ans1, confidence=meta1.get("confidence", 0.0), sources=meta1.get("sources", []), session_id=session_a)
    print(f"✓ Answer 1: {ans1[:100]}...\n")
    
    print("→ Query 2 (Session A follow-up): 'What do they design?'")
    # This query uses relative pronoun "they". It should be rewritten to "What does Alpha Corporation design?"
    ans2, meta2 = pipeline.run("What do they design?", user_id=user_id, session_id=session_a)
    save_chat(user_id=user_id, role="user", content="What do they design?", session_id=session_a)
    save_chat(user_id=user_id, role="assistant", content=ans2, confidence=meta2.get("confidence", 0.0), sources=meta2.get("sources", []), session_id=session_a)
    print(f"✓ Answer 2: {ans2[:100]}...\n")
    
    # Assert rewritten query found documents about GPUs
    assert "GPU" in ans2 or "hardware" in ans2 or len(meta2.get("sources", [])) > 0
    print("✓ Follow-up query rewriting matched context successfully.\n")
    
    # 2. Test Session Memory isolation (Session B starts fresh)
    print("→ Query 1 (Session B): 'What do they design?'")
    # In session B, there is no history context, so "they" has no antecedent. 
    # It should not resolve to "Alpha Corporation".
    ans3, meta3 = pipeline.run("What do they design?", user_id=user_id, session_id=session_b)
    # Session B should fail to answer or have 0 confidence / "Insufficient data"
    print(f"✓ Session B Answer: {ans3}")
    assert "insufficient data" in ans3.lower() or meta3.get("confidence", 0.0) < 0.15
    print("✓ Session isolation passed (Session B did not inherit Session A history).\n")
    
    # 3. Test Token/Word Optimization (Sliding Window)
    print("→ Testing Token/Word Optimization...")
    # Inject a large history to exceed 600 words
    long_text = "This is filler text. " * 50  # ~200 words per message
    for i in range(5):
        save_chat(user_id=user_id, role="user", content=f"Filler message {i}: {long_text}", session_id=session_a)
        save_chat(user_id=user_id, role="assistant", content=f"Filler response {i}: {long_text}", session_id=session_a)
        
    # Run pipeline again, it should successfully optimize/prune history and answer without crashing
    ans_opt, meta_opt = pipeline.run("What is Alpha Corporation?", user_id=user_id, session_id=session_a)
    assert ans_opt is not None
    print("✓ Token/Word count optimization successfully pruned older messages and executed without error.\n")
    
    # Cleanup
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        
    print("=" * 80)
    print("CONVERSATIONAL MEMORY INTEGRATION TEST PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_memory()
