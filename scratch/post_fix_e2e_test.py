"""
End-to-end test: PDF -> DOCX -> YouTube query pipeline after fixes.
Run with: .venv\Scripts\python.exe -X utf8 scratch/post_fix_e2e_test.py
"""
import sys, time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR  = ROOT_DIR / "src"
for p in [str(SRC_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

from core.config import FAISS_INDEX_PATH, CHUNKS_PATH, BM25_INDEX_PATH, EMBEDDING_MODEL_NAME, INDEX_DIR
from core.embedding_manager import EmbeddingManager
from retrieval.keyword_search import KeywordSearch
from retrieval.retriever import HybridRetriever
from llm.generator import LLMGenerator
from core.pipeline import RAGPipeline
from ingestion.document_processor import DocumentProcessor

DIV = "=" * 72
MINI = "-" * 50

def header(title: str) -> None:
    print(f"\n{DIV}\n{title}\n{DIV}")

def mini(title: str) -> None:
    print(f"\n{MINI}\n{title}\n{MINI}")

# ── Load shared components ────────────────────────────────────────────
header("POST-FIX END-TO-END TEST")
print(f"INDEX_DIR   : {INDEX_DIR}")
print(f"FAISS path  : {FAISS_INDEX_PATH}")
print(f"FAISS exists: {FAISS_INDEX_PATH.exists()}")

em  = EmbeddingManager(index_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH)
ks  = KeywordSearch(BM25_INDEX_PATH)
llm = LLMGenerator()
retriever = HybridRetriever(em, ks)
pipeline  = RAGPipeline(retriever, llm)

print(f"Chunks in FAISS: {len(em.chunks)}")
print(f"FAISS ntotal   : {em.index.ntotal if em.index else 0}")

# ── Determine which doc sources are in the index ─────────────────────
sources = {}
for c in em.chunks:
    src = c.get("metadata", {}).get("source", "?")
    sources[src] = sources.get(src, 0) + 1

docx_source = next((s for s in sources if "Transformer" in s or ".docx" in s.lower()), None)
pdf_source  = next((s for s in sources if ".pdf" in s.lower()), None)
yt_source   = next((s for s in sources if "youtube.com" in s or "youtu.be" in s), None)
test_uid    = None
for c in em.chunks:
    if c.get("metadata", {}).get("source") == (docx_source or pdf_source or yt_source):
        test_uid = c.get("metadata", {}).get("user_id")
        break

print(f"\nDocuments in index:")
for src, cnt in list(sources.items())[:8]:
    print(f"  {src}: {cnt} chunks")
print(f"\nSelected for test:")
print(f"  DOCX  source : {docx_source!r}")
print(f"  PDF   source : {pdf_source!r}")
print(f"  YT    source : {yt_source!r}")
print(f"  user_id      : {test_uid!r}")

# ── Helper: run query and print result ────────────────────────────────
def run_test(label: str, query: str, user_id: str | None = None, custom_pipeline: RAGPipeline | None = None):
    mini(label)
    print(f"  QUERY   : {query}")
    print(f"  user_id : {user_id!r}")
    active_pipeline = custom_pipeline or pipeline
    t0 = time.perf_counter()
    try:
        answer, meta = active_pipeline.run(query, user_id=user_id, session_id="e2e_test")
    except Exception as e:
        print(f"  RESULT  : FAIL")
        print(f"  ERROR   : {type(e).__name__}: {e}")
        return

    chunks   = meta.get("sources", [])
    conf     = meta.get("confidence", 0.0)
    latency  = meta.get("latency", 0.0)
    rejected = answer == "Insufficient data"

    print(f"  RESULT  : {'FAIL (Insufficient data)' if rejected else 'PASS'}")
    print(f"  Chunks retrieved  : {len(chunks)}")
    if chunks:
        top = chunks[0]
        faiss_l2 = top.get("score", 0.0)
        rerank   = top.get("rerank_score", 0.0)
        raw_ce   = top.get("raw_rerank_score", 0.0)
        src      = top.get("metadata", {}).get("source", "?")
        print(f"  Top chunk source  : {src!r}")
        print(f"  FAISS L2 dist     : {faiss_l2:.4f}")
        print(f"  CE raw logit      : {raw_ce:.4f}")
        print(f"  Reranker score    : {rerank:.6f}")
    print(f"  CONFIDENCE        : {conf:.6f}  ({conf*100:.2f}%)")
    print(f"  THRESHOLD         : 0.15  (15.00%)")
    print(f"  GATE              : {'REJECTED' if rejected else 'PASSED'}")
    print(f"  LATENCY           : {latency:.2f}s")
    if not rejected:
        print(f"  GROQ ANSWER       : {answer[:400]!r}...")

# ── TEST 1: DOCX ──────────────────────────────────────────────────────
if docx_source:
    run_test(
        "TEST 1: DOCX Query (Basics of Transformer.docx)",
        "What is a Transformer model and how does self-attention work?",
        user_id=test_uid
    )
else:
    mini("TEST 1: DOCX")
    print("  SKIP — no .docx file in index")

# ── TEST 2: PDF ───────────────────────────────────────────────────────
sample_pdf_path = ROOT_DIR / "SampleFile.pdf"
if not pdf_source and sample_pdf_path.exists():
    print("\n  Ingesting SampleFile.pdf for PDF validation...")
    processor = DocumentProcessor()
    pdf_chunks = processor.process_document(str(sample_pdf_path), user_id=test_uid or "pdf_test_user")
    em.add_chunks(pdf_chunks, save=True)
    ks.add_chunks(pdf_chunks)
    pipeline.retriever = HybridRetriever(em, ks)
    pdf_source = "SampleFile.pdf"

if pdf_source:
    run_test(
        "TEST 2: PDF Query (SampleFile.pdf)",
        "What projects has Ruthvik Thalapaneni worked on?",
        user_id=test_uid or "pdf_test_user"
    )
else:
    mini("TEST 2: PDF")
    print("  SKIP — no .pdf file in index")

# ── TEST 3: YouTube transcript ingestion + query ───────────────────────
mini("TEST 3: YouTube Transcript Ingestion + Query")
YT_URL  = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
YT_USER = "e2e_test_user"
print(f"  URL     : {YT_URL}")
print(f"  user_id : {YT_USER!r}")

try:
    from connectors.youtube import YouTubeConnector
    t0 = time.perf_counter()
    yt_connector = YouTubeConnector(urls=[YT_URL])
    yt_docs = yt_connector.fetch_documents()
    print(f"  Fetch time    : {time.perf_counter()-t0:.2f}s")
    print(f"  Docs returned : {len(yt_docs)}")

    if yt_docs:
        doc = yt_docs[0]
        raw = doc.get("raw_data", b"")
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        print(f"  Transcript bytes  : {len(raw)}")

        processor = DocumentProcessor()
        chunks = processor.process_raw_data(
            raw_data=raw,
            source_name=doc["source"],
            extension=doc["extension"],
            user_id=YT_USER,
            extra_metadata=doc.get("metadata", {})
        )
        print(f"  Chunks created    : {len(chunks)}")

        # Add to a FRESH in-memory embedding manager (don't pollute the main index)
        from core.embedding_manager import EmbeddingManager as EM2
        em2 = EM2()
        em2.add_chunks(chunks, save=False)
        print(f"  FAISS ntotal (yt) : {em2.index.ntotal if em2.index else 0}")

        from retrieval.keyword_search import KeywordSearch as KS2
        yt_bm25_path = INDEX_DIR / "bm25_test_yt.pkl"
        if yt_bm25_path.exists():
            yt_bm25_path.unlink()
        ks2 = KS2(yt_bm25_path)
        ks2.add_chunks(chunks)

        retriever2 = HybridRetriever(em2, ks2)
        pipeline2  = RAGPipeline(retriever2, llm)

        run_test(
            "TEST 3b: Query against YouTube transcript",
            "What is this song about and what are the promises made in Never Gonna Give You Up?",
            user_id=YT_USER,
            custom_pipeline=pipeline2
        )
    else:
        print("  RESULT  : FAIL — fetch_documents() returned empty list (unexpected, should raise)")

except RuntimeError as e:
    print(f"  RESULT  : FAIL (RuntimeError surfaced correctly)")
    print(f"  ERROR   : {e}")
    print("  NOTE    : On deployment this is expected — YouTube blocks datacenter IPs.")
    print("            The error is now VISIBLE to the user via st.error() instead of being hidden.")
except Exception as e:
    print(f"  RESULT  : FAIL (unexpected exception)")
    print(f"  ERROR   : {type(e).__name__}: {e}")

# ── SUMMARY ───────────────────────────────────────────────────────────
header("SUMMARY")
print(f"INDEX_DIR now  : {INDEX_DIR}")
print(f"On Render      : /app/data/embeddings/  <- ON persistent disk (mountPath: /app/data)")
print(f"On Docker local: ./data/embeddings/     <- covered by ./data:/app/data volume")
print(f"YouTube errors : now raised as RuntimeError, visible via st.error() in UI")
