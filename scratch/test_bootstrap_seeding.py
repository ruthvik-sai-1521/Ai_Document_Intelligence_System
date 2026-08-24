"""
Test Bootstrap Seeding Scenario:
Simulates a fresh cloud deployment cold boot (0 vectors, empty storage)
and validates that SampleFile.pdf is automatically detected, seeded, indexed,
and queried end-to-end with Groq response generation.
"""
import sys
from pathlib import Path
import shutil
import tempfile

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for p in [str(SRC_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

from core.logger import setup_logger
logger = setup_logger("test_bootstrap")

from core.embedding_manager import EmbeddingManager
from retrieval.keyword_search import KeywordSearch
from retrieval.retriever import HybridRetriever
from llm.generator import LLMGenerator
from core.pipeline import RAGPipeline
from ingestion.document_processor import DocumentProcessor

print("=" * 70)
print("TESTING FRESH APPLICATION BOOTSTRAP SEEDING (CLOUD DEPLOYMENT SIMULATION)")
print("=" * 70)

# Create a temporary empty directory to simulate a fresh cloud filesystem
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)
    faiss_path = tmp_path / "faiss_index.bin"
    chunks_path = tmp_path / "chunks.pkl"
    bm25_path = tmp_path / "bm25_index.pkl"

    print(f"\n1. Starting with fresh empty paths in: {tmp_path}")
    print(f"   faiss_path exists? : {faiss_path.exists()} (Expected: False)")

    # 1. Initialize components with empty paths (exactly as load_models does on cloud boot)
    em = EmbeddingManager(model_name="all-MiniLM-L6-v2", index_path=faiss_path, chunks_path=chunks_path)
    ks = KeywordSearch(bm25_path)
    llm = LLMGenerator()

    print(f"   Initial FAISS ntotal : {em.index.ntotal if em.index else 0} (Expected: 0)")
    print(f"   Initial chunks count : {len(em.chunks)} (Expected: 0)")

    # 2. Run the bootstrap seeding block
    if em.index is None or em.index.ntotal == 0 or not em.chunks:
        logger.info("[BOOTSTRAP] Starting...")
        sample_pdf = ROOT_DIR / "SampleFile.pdf"
        if sample_pdf.exists():
            try:
                logger.info(f"[BOOTSTRAP] Seed document found: {sample_pdf.name}")
                processor = DocumentProcessor()
                seed_chunks = processor.process_document(str(sample_pdf), user_id=None)
                if seed_chunks:
                    logger.info(f"[BOOTSTRAP] Created {len(seed_chunks)} chunks")
                    em.add_chunks(seed_chunks, save=True)
                    logger.info(f"[BOOTSTRAP] FAISS now contains {em.index.ntotal if em.index else len(seed_chunks)} vectors")
                    ks.add_chunks(seed_chunks)
                    logger.info("[BOOTSTRAP] BM25 initialized")
                    logger.info("[BOOTSTRAP] Completed successfully")
                else:
                    logger.warning("[BOOTSTRAP] No chunks generated from seed document.")
            except Exception as e:
                logger.warning(f"[BOOTSTRAP] Failed to process seed document: {e}")
        else:
            logger.warning(f"[BOOTSTRAP] Seed document not found at {sample_pdf}. Continuing with empty vector store.")

    # 3. Verify vector store & BM25 are now populated
    print(f"\n2. Post-Bootstrap Verification:")
    print(f"   FAISS ntotal         : {em.index.ntotal if em.index else 0} (Expected: > 0)")
    print(f"   Chunks loaded        : {len(em.chunks)}")
    print(f"   BM25 corpus size     : {len(ks.chunks)}")
    print(f"   FAISS file created   : {faiss_path.exists()}")
    print(f"   Chunks file created  : {chunks_path.exists()}")
    print(f"   BM25 file created    : {bm25_path.exists()}")

    assert em.index is not None and em.index.ntotal > 0, "FAISS index should not be empty after bootstrap!"
    assert len(em.chunks) > 0, "Chunks list should not be empty after bootstrap!"
    assert len(ks.chunks) > 0, "BM25 index should not be empty after bootstrap!"

    # 4. Run end-to-end RAG query against the bootstrapped knowledge
    print(f"\n3. Running End-to-End Query against Bootstrapped Index:")
    retriever = HybridRetriever(em, ks)
    pipeline = RAGPipeline(retriever, llm)

    query = "What projects has Ruthvik Thalapaneni worked on?"
    print(f"   Query: '{query}'")

    answer, meta = pipeline.run(query, user_id=None, session_id="bootstrap_test")

    print(f"\n4. Pipeline Results:")
    print(f"   Confidence score     : {meta.get('confidence', 0.0):.4f} ({meta.get('confidence', 0.0)*100:.1f}%)")
    print(f"   Retrieved chunks     : {len(meta.get('sources', []))}")
    if meta.get('sources'):
        top = meta['sources'][0]
        print(f"   Top chunk source     : {top.get('metadata', {}).get('source')}")
        print(f"   Rerank score         : {top.get('rerank_score', 0.0):.6f}")
    print(f"   Rejection status     : {'REJECTED' if answer == 'Insufficient data' else 'PASSED'}")
    print(f"\n5. Groq Generated Answer:\n{answer}")

    assert answer != "Insufficient data", "Query should not be rejected on bootstrapped index!"
    print("\n" + "=" * 70)
    print("BOOTSTRAP SEEDING TEST PASSED SUCCESSFULLY!")
    print("=" * 70)
