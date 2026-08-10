import sys
from pathlib import Path
import os
import sqlite3

# Add project root to sys.path
PROJECT_ROOT = Path(r"d:\Projects\Ai_Document_Intelligence_System")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
from src.connectors.db import DatabaseConnector
from src.ingestion.document_processor import DocumentProcessor
from src.core.embedding_manager import EmbeddingManager
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.retriever import HybridRetriever
from src.llm.generator import LLMGenerator
from src.core.pipeline import RAGPipeline
from src.core.config import INDEX_DIR

TEST_DB_PATH = PROJECT_ROOT / "scratch" / "test_products.db"
TEST_FAISS_PATH = INDEX_DIR / "test_db_faiss.bin"
TEST_CHUNKS_PATH = INDEX_DIR / "test_db_chunks.pkl"
TEST_BM25_PATH = INDEX_DIR / "test_db_bm25.pkl"

def create_mock_db():
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
        
    conn = sqlite3.connect(str(TEST_DB_PATH))
    cursor = conn.cursor()
    
    # Create products table with primary key
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Products (
            ProductID TEXT PRIMARY KEY,
            Name TEXT,
            Price REAL,
            Category TEXT
        )
    """)
    
    products = [
        ("P1001", "Quantum Neural Processor", 1499.99, "Hardware"),
        ("P1002", "Holographic Storage Drive", 450.00, "Storage"),
        ("P1003", "Bio-mimetic Interface Core", 2999.00, "Interface")
    ]
    cursor.executemany("INSERT INTO Products (ProductID, Name, Price, Category) VALUES (?, ?, ?, ?)", products)
    conn.commit()
    conn.close()
    print(f"✓ Mock database created at {TEST_DB_PATH}")

def cleanup():
    for p in [TEST_DB_PATH, TEST_FAISS_PATH, TEST_CHUNKS_PATH, TEST_BM25_PATH]:
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                pass

def test_structured_retrieval():
    print("=" * 80)
    print("STARTING STRUCTURED DATA RAG INTEGRATION TEST")
    print("=" * 80)

    cleanup()
    create_mock_db()

    # 1. Initialize connector
    db_config = {"db_path": str(TEST_DB_PATH)}
    connector = DatabaseConnector(db_type="sqlite", config=db_config)
    
    # 2. Fetch documents
    db_docs = connector.fetch_documents()
    assert len(db_docs) == 1, "Should have retrieved 1 document representation for the table"
    doc = db_docs[0]
    
    assert doc["extension"] == ".json", "Database representation extension should be .json"
    assert doc["metadata"]["table_name"] == "Products"
    assert "sqlite:///" in doc["metadata"]["source"]
    print("✓ Connector extraction verified successfully")

    # 3. Process & Chunk
    processor = DocumentProcessor()
    chunks = processor.process_raw_data(
        raw_data=doc["raw_data"],
        source_name=doc["source"],
        extension=doc["extension"],
        extra_metadata=doc.get("metadata", {})
    )
    
    # There are 3 rows in mock products table. Each row should yield 1 semantic chunk page.
    assert len(chunks) == 3, f"Expected 3 semantic row chunks, got {len(chunks)}"
    print("✓ Row-by-row semantic chunking layout verified:")
    print("-" * 60)
    print(chunks[0]["text"])
    print("-" * 60)
    
    # Check that metadata elements are properly attached
    first_chunk_meta = chunks[0]["metadata"]
    assert first_chunk_meta["source_type"] == "structured_row"
    assert first_chunk_meta["primary_key_column"] == "ProductID"
    assert first_chunk_meta["primary_key_value"] in ["P1001", "P1002", "P1003"]
    assert "Name" in first_chunk_meta["columns"]
    print("✓ Row-level metadata attributes verified successfully")

    # 4. Embed & Index
    emb_mgr = EmbeddingManager(index_path=TEST_FAISS_PATH, chunks_path=TEST_CHUNKS_PATH)
    kw_search = KeywordSearch(index_path=TEST_BM25_PATH)
    
    emb_mgr.add_chunks(chunks, save=True)
    kw_search.add_chunks(chunks)
    
    # 5. Hybrid Retrieval
    retriever = HybridRetriever(emb_mgr, kw_search)
    llm = LLMGenerator()
    pipeline = RAGPipeline(retriever, llm)
    
    # Search specifically for Holographic Drive
    query = "How much does the Holographic Storage Drive cost and what is its product ID?"
    retrieved = retriever.retrieve(query, top_k=2)
    
    assert len(retrieved) > 0
    # Check that the holographic drive chunk is the highest scorer or matching source
    match_source = False
    for r in retrieved:
        if "Holographic Storage Drive" in r["text"]:
            match_source = True
            print(f"✓ Retrieved chunk: Score={r.get('score', 0):.4f} | PK={r['metadata'].get('primary_key_value')}")
            
    assert match_source, "Holographic storage drive row should be retrieved"
    
    # Run pipeline & verify generation
    answer, meta = pipeline.run(query)
    print("\n--- LLM Final Answer ---")
    print(answer)
    print("------------------------")
    
    assert "450" in answer, "Answer should extract the price: $450"
    assert "P1002" in answer, "Answer should retrieve primary key ProductID: P1002"
    
    # Verify citations format
    print("\n--- Answer Citations ---")
    for s in meta.get("sources", []):
        print(f"- Source: {s.get('metadata', {}).get('source')} | PK: {s.get('metadata', {}).get('primary_key_value')}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUCCESSFUL: STRUCTURED DATA SEMANTIC PIPELINE OPERATIONAL!")
    print("=" * 80)
    cleanup()

if __name__ == "__main__":
    test_structured_retrieval()
