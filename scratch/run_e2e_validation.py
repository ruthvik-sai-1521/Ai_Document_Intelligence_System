import sys
from pathlib import Path
import os

# Add project root and src directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Configure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import time
from typing import List, Dict, Any

from src.ingestion.document_processor import DocumentProcessor
from src.core.embedding_manager import EmbeddingManager
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.retriever import HybridRetriever
from src.llm.generator import LLMGenerator
from src.core.pipeline import RAGPipeline
from src.core.config import INDEX_DIR, DATA_DIR
from src.connectors.web import WebConnector
from src.connectors.github import GitHubConnector
from src.connectors.youtube import YouTubeConnector

# Define temporary verification paths
TEST_FAISS_PATH = INDEX_DIR / "e2e_val_faiss.bin"
TEST_CHUNKS_PATH = INDEX_DIR / "e2e_val_chunks.pkl"
TEST_BM25_PATH = INDEX_DIR / "e2e_val_bm25.pkl"

def cleanup_indexes():
    for p in [TEST_FAISS_PATH, TEST_CHUNKS_PATH, TEST_BM25_PATH]:
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                print(f"Cleanup error for {p}: {e}")

def run_e2e_validation():
    print("=" * 80)
    print("STARTING FULL END-TO-END RAG VALIDATION RUN")
    print("=" * 80)

    cleanup_indexes()

    emb_mgr = EmbeddingManager(index_path=TEST_FAISS_PATH, chunks_path=TEST_CHUNKS_PATH)
    kw_search = KeywordSearch(index_path=TEST_BM25_PATH)
    retriever = HybridRetriever(emb_mgr, kw_search)
    llm = LLMGenerator()
    pipeline = RAGPipeline(retriever, llm, confidence_threshold=0.01)

    processor = DocumentProcessor()
    
    report = {
        "individual_sources": {},
        "mixed_sources": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 1. Verify Individual Sources
    fixtures_dir = DATA_DIR / "e2e_test_fixtures"
    
    # Format: (Key, filename/args, type, extension, query, expected_keywords)
    source_tests = [
        ("PDF", "legal_contract.pdf", "file", ".pdf", "What is the document about?", ["contract", "agreement", "parties"]),
        ("TXT", "hr_policy.txt", "file", ".txt", "What is the vacation policy?", ["vacation", "days", "leave"]),
        ("DOCX", "finance_strategy.docx", "file", ".docx", "What is the financial strategy?", ["finance", "revenue", "growth"]),
        ("PPTX", "product_roadmap.pptx", "file", ".pptx", "What is the product roadmap?", ["product", "roadmap", "timeline"]),
        ("XLSX", "sales_report.xlsx", "file", ".xlsx", "What is the sales report?", ["sales", "q1", "revenue"]),
        ("CSV", "employee_directory.csv", "file", ".csv", "Who are the employees?", ["john", "jane", "employees"]),
        ("Markdown", "architecture.md", "file", ".md", "What is the architecture?", ["architecture", "rag", "system"]),
        ("HTML", "support_kb.html", "file", ".html", "What is the support KB?", ["support", "kb", "help"]),
        ("Images (OCR)", "receipt_ocr.png", "file", ".png", "What is the receipt about?", ["receipt", "total", "amount"]),
        ("Scanned PDF", "scanned_doc.pdf", "file", ".pdf", "What is in the scanned document?", ["document", "scanned", "text"])
    ]

    for key, name, src_type, ext, query, keywords in source_tests:
        print(f"\n--- Testing Source: {key} ---")
        try:
            file_path = fixtures_dir / name
            if not file_path.exists():
                print(f"ERROR: Fixture {file_path} not found.")
                report["individual_sources"][key] = {"status": "FAIL", "reason": f"Fixture not found at {file_path}"}
                continue

            # Process & Chunk
            chunks = processor.process_document(str(file_path))
            print(f"✓ Ingested & Chunked: Generated {len(chunks)} chunk(s)")
            
            # Embed & Index
            emb_mgr.add_chunks(chunks, save=True)
            kw_search.add_chunks(chunks)
            print("✓ Embedded & Indexed successfully")

            # Retrieval
            retrieved = retriever.retrieve(query, top_k=2)
            print(f"✓ Retrieved {len(retrieved)} chunk(s)")
            
            # Generate Answer
            answer, meta = pipeline.run(query)
            print(f"✓ Answer generated (Confidence: {meta.get('confidence'):.4f})")
            
            has_citation = any(key.lower() in str(s.get("metadata", {}).get("source", "")).lower() for s in meta.get("sources", []))
            
            report["individual_sources"][key] = {
                "status": "PASS",
                "chunks": len(chunks),
                "retrieved": len(retrieved),
                "confidence": meta.get("confidence"),
                "has_citation": has_citation,
                "answer_preview": answer[:150]
            }
        except Exception as e:
            print(f"❌ Error during {key} test: {e}")
            report["individual_sources"][key] = {"status": "FAIL", "reason": str(e)}

    # 11. Website Ingestion
    print("\n--- Testing Source: Website Ingestion ---")
    try:
        web_connector = WebConnector(urls=["https://example.com"], max_depth=1)
        docs = web_connector.fetch_documents()
        if not docs:
            raise RuntimeError("Crawl failed to fetch any pages")
        
        web_chunks = []
        for doc in docs:
            chunks = processor.process_raw_data(
                raw_data=doc["raw_data"],
                source_name=doc["source"],
                extension=doc["extension"]
            )
            web_chunks.extend(chunks)
            
        print(f"✓ Ingested & Chunked: Generated {len(web_chunks)} chunk(s)")
        emb_mgr.add_chunks(web_chunks, save=True)
        kw_search.add_chunks(web_chunks)

        query = "What is the Example Domain description?"
        answer, meta = pipeline.run(query)
        has_citation = any("example.com" in str(s.get("metadata", {}).get("source", "")).lower() for s in meta.get("sources", []))

        report["individual_sources"]["Website"] = {
            "status": "PASS",
            "chunks": len(web_chunks),
            "confidence": meta.get("confidence"),
            "has_citation": has_citation,
            "answer_preview": answer[:150]
        }
    except Exception as e:
        print(f"❌ Error during Website test: {e}")
        report["individual_sources"]["Website"] = {"status": "FAIL", "reason": str(e)}

    # 12. GitHub Repository Ingestion
    print("\n--- Testing Source: GitHub Ingestion ---")
    try:
        gh_connector = GitHubConnector(repo_url="octocat/Spoon-Knife", branch="main")
        gh_docs = gh_connector.fetch_documents()
        if not gh_docs:
            raise RuntimeError("GitHub zip download or parsing failed")

        gh_chunks = []
        # Limit to first 3 docs to keep speed up
        for doc in gh_docs[:3]:
            chunks = processor.process_raw_data(
                raw_data=doc["raw_data"],
                source_name=doc["source"],
                extension=doc["extension"],
                extra_metadata=doc.get("metadata", {})
            )
            gh_chunks.extend(chunks)

        print(f"✓ Ingested & Chunked: Generated {len(gh_chunks)} chunk(s)")
        emb_mgr.add_chunks(gh_chunks, save=True)
        kw_search.add_chunks(gh_chunks)

        query = "Show repository details Spoon Knife"
        answer, meta = pipeline.run(query)
        has_citation = any("github.com" in str(s.get("metadata", {}).get("source", "")).lower() for s in meta.get("sources", []))

        report["individual_sources"]["GitHub"] = {
            "status": "PASS",
            "chunks": len(gh_chunks),
            "confidence": meta.get("confidence"),
            "has_citation": has_citation,
            "answer_preview": answer[:150]
        }
    except Exception as e:
        print(f"❌ Error during GitHub test: {e}")
        report["individual_sources"]["GitHub"] = {"status": "FAIL", "reason": str(e)}

    # 13. YouTube Ingestion
    print("\n--- Testing Source: YouTube Ingestion ---")
    try:
        # Fetch Rick Astley public video which has english captions enabled
        yt_connector = YouTubeConnector(urls=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        yt_docs = yt_connector.fetch_documents()
        if not yt_docs:
            raise RuntimeError("YouTube transcript fetch failed")

        yt_chunks = []
        for doc in yt_docs:
            chunks = processor.process_raw_data(
                raw_data=doc["raw_data"],
                source_name=doc["source"],
                extension=doc["extension"],
                extra_metadata=doc.get("metadata", {})
            )
            yt_chunks.extend(chunks)

        print(f"✓ Ingested & Chunked: Generated {len(yt_chunks)} chunk(s)")
        emb_mgr.add_chunks(yt_chunks, save=True)
        kw_search.add_chunks(yt_chunks)

        query = "What are the lyrics of the song in the video?"
        answer, meta = pipeline.run(query)
        has_citation = any("youtube.com" in str(s.get("metadata", {}).get("source", "")).lower() for s in meta.get("sources", []))

        report["individual_sources"]["YouTube"] = {
            "status": "PASS",
            "chunks": len(yt_chunks),
            "confidence": meta.get("confidence"),
            "has_citation": has_citation,
            "answer_preview": answer[:150]
        }
    except Exception as e:
        print(f"❌ Error during YouTube test: {e}")
        report["individual_sources"]["YouTube"] = {"status": "FAIL", "reason": str(e)}


    # 2. Mixed-Source Retrieval Scenarios
    mixed_scenarios = [
        ("PDF + Website", "Compare legal contract terms and example domain properties", ["legal_contract.pdf", "example.com"]),
        ("GitHub + YouTube", "Relate Spoon Knife project code details to the song lyrics video", ["github.com", "youtube.com"]),
        ("OCR + DOCX", "Combine the financial strategy details with the receipt amount", ["receipt_ocr.png", "finance_strategy.docx"]),
        ("CSV + Website", "Link employee directory listings to example domain features", ["employee_directory.csv", "example.com"]),
        ("YouTube + PDF", "Relate legal contract guidelines to YouTube video timestamps", ["dQw4w9WgXcQ", "legal_contract.pdf"])
    ]

    print("\n" + "=" * 80)
    print("TESTING MIXED-SOURCE RETRIEVAL SCENARIOS")
    print("=" * 80)

    for key, query, expected_sources in mixed_scenarios:
        print(f"\n--- Scenario: {key} ---")
        try:
            answer, meta = pipeline.run(query)
            retrieved_sources = [s.get("metadata", {}).get("source", "") for s in meta.get("sources", [])]
            
            matched = []
            for expected in expected_sources:
                found = any(expected.lower() in src.lower() for src in retrieved_sources)
                matched.append(found)
                print(f"  - Target '{expected}' found in citations: {found}")

            all_matched = all(matched)
            print(f"✓ Scenario result: {'PASS' if all_matched else 'PARTIAL/FAIL'} (Confidence: {meta.get('confidence'):.4f})")
            
            report["mixed_sources"][key] = {
                "status": "PASS" if all_matched else "PARTIAL",
                "confidence": meta.get("confidence"),
                "matched_targets": matched,
                "answer_preview": answer[:150],
                "retrieved_list": retrieved_sources
            }
        except Exception as e:
            print(f"❌ Error during Mixed {key} test: {e}")
            report["mixed_sources"][key] = {"status": "FAIL", "reason": str(e)}

    # Save validation metrics JSON
    with open(PROJECT_ROOT / "scratch" / "e2e_validation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    cleanup_indexes()
    print("\n" + "=" * 80)
    print("END-TO-END RAG VALIDATION RUN COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    run_e2e_validation()
