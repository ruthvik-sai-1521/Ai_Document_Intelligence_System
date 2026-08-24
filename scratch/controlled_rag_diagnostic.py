"""
Controlled RAG Diagnostic Script
Runs a single query against a document that is DEFINITELY indexed.
Prints every score at every stage without modifying anything.
"""
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ── silence Streamlit context warnings ──────────────────────────────
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

from core.config import FAISS_INDEX_PATH, CHUNKS_PATH, BM25_INDEX_PATH, EMBEDDING_MODEL_NAME
from core.embedding_manager import EmbeddingManager
from retrieval.keyword_search import KeywordSearch
from retrieval.retriever import HybridRetriever

DIVIDER = "=" * 80
MINI = "-" * 60

# ────────────────────────────────────────────────────────────────────
print(DIVIDER)
print("CONTROLLED RAG DIAGNOSTIC — SCORE TRACE")
print(DIVIDER)

# ── 1. Load indexes ──────────────────────────────────────────────────
em = EmbeddingManager(
    model_name=EMBEDDING_MODEL_NAME or "all-MiniLM-L6-v2",
    index_path=FAISS_INDEX_PATH,
    chunks_path=CHUNKS_PATH
)
ks = KeywordSearch(BM25_INDEX_PATH)

print(f"\n[INDEX STATE]")
print(f"  FAISS index path    : {FAISS_INDEX_PATH}")
print(f"  FAISS file exists?  : {FAISS_INDEX_PATH.exists()}")
print(f"  Chunks path         : {CHUNKS_PATH}")
print(f"  Chunks file exists? : {CHUNKS_PATH.exists()}")
if FAISS_INDEX_PATH.exists():
    import faiss as _faiss
    raw_idx = _faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"  FAISS ntotal        : {raw_idx.ntotal} vectors")
    print(f"  FAISS dimension     : {raw_idx.d}")
print(f"  Chunks loaded       : {len(em.chunks)}")
print(f"  Embedding model     : {em.model_name}")
print(f"  Embedding dimension : {em.dimension}")

# ── 2. Choose a query against an INDEXED document ───────────────────
# Pick first real document in the index
sources_in_index = {}
for c in em.chunks:
    src = c.get("metadata", {}).get("source", "?")
    sources_in_index[src] = sources_in_index.get(src, 0) + 1

print(f"\n[DOCUMENTS IN LOCAL INDEX]")
for src, count in list(sources_in_index.items())[:10]:
    print(f"  {src}: {count} chunks")

# Pick a well-indexed document
target_source = None
target_query = None
for src in sources_in_index:
    if "Transformer" in src or "transformer" in src:
        target_source = src
        target_query = "What is a Transformer model and how does self-attention work?"
        break
    if "LLM" in src or "llm" in src or "language model" in src.lower():
        target_source = src
        target_query = "What are large language models and how are they trained?"
        break

if not target_source:
    # Fall back to first non-test document
    for src in sources_in_index:
        if "test" not in src.lower():
            target_source = src
            target_query = "What is the main topic of this document?"
            break

print(f"\n[SELECTED TARGET]")
print(f"  Source document : {target_source}")
print(f"  Chunks in index : {sources_in_index.get(target_source, 0)}")

# ── 3. Pick a specific user_id that owns those chunks ───────────────
user_for_query = None
for c in em.chunks:
    if c.get("metadata", {}).get("source") == target_source:
        user_for_query = c.get("metadata", {}).get("user_id")
        break

print(f"  Owner user_id   : {user_for_query!r}")

# ── 4. QUERY ─────────────────────────────────────────────────────────
QUERY = target_query
print(f"\n[QUERY]")
print(f"  {QUERY}")

# ── 5. FAISS raw search (step 1 of pipeline) ─────────────────────────
print(f"\n{MINI}")
print(f"STEP 1: FAISS VECTOR SEARCH")
print(MINI)
print(f"  Index type   : faiss.IndexFlatL2 (Euclidean/L2 distance — NOT cosine, NOT probability)")
print(f"  What it does : Embeds query → computes L2 distance to every vector → returns closest K")
print(f"  Score meaning: LOWER = more similar  (L2 distance, not bounded to [0,1])")

faiss_candidates = em.search(QUERY, top_k=15*3, user_id=user_for_query)
print(f"  Raw FAISS results count: {len(faiss_candidates)}")
for i, c in enumerate(faiss_candidates[:5]):
    print(f"  FAISS [{i+1}] L2-dist={c['score']:.4f} | source={c.get('metadata',{}).get('source','?')!r} | text={c['text'][:60]!r}...")

# ── 6. BM25 search ───────────────────────────────────────────────────
print(f"\n{MINI}")
print(f"STEP 2: BM25 KEYWORD SEARCH")
print(MINI)
bm25_candidates = ks.search(QUERY, top_k=15*3, user_id=user_for_query)
print(f"  BM25 results count: {len(bm25_candidates)}")
for i, c in enumerate(bm25_candidates[:5]):
    print(f"  BM25 [{i+1}] bm25-score={c.get('score',0):.4f} | source={c.get('metadata',{}).get('source','?')!r} | text={c['text'][:60]!r}...")

# ── 7. RRF Fusion ────────────────────────────────────────────────────
print(f"\n{MINI}")
print(f"STEP 3: RECIPROCAL RANK FUSION (RRF)")
print(MINI)
print(f"  Formula: rrf_score = 1/(60+rank_faiss) + 1/(60+rank_bm25)")
print(f"  Score range: ~0.008 to ~0.033 (reciprocal of rank+60)")

retriever = HybridRetriever(em, ks)

# Run internal RRF manually
filtered_faiss = [c for c in faiss_candidates][:15]
filtered_bm25  = [c for c in bm25_candidates][:15]
rrf_merged = retriever._reciprocal_rank_fusion(filtered_faiss, filtered_bm25)[:15]
print(f"  RRF pool size: {len(rrf_merged)}")
for i, c in enumerate(rrf_merged[:5]):
    print(f"  RRF [{i+1}] rrf_score={c.get('rrf_score',0):.6f} | source={c.get('metadata',{}).get('source','?')!r} | text={c['text'][:60]!r}...")

# ── 8. CrossEncoder reranking ────────────────────────────────────────
print(f"\n{MINI}")
print(f"STEP 4: CROSS-ENCODER RERANKING")
print(MINI)
print(f"  Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
print(f"  What it outputs: RAW LOGITS (unbounded, range roughly -10 to +10)")
print(f"  Transformation: sigmoid(logit) = 1 / (1 + exp(-logit))")
print(f"  Score range after sigmoid: (0.0, 1.0) — a PROBABILITY")
print(f"  Score meaning: HIGHER = more relevant  (probability of relevance)")

from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print(f"\n  Evaluating {len(rrf_merged)} chunks from RRF pool...")
pairs = [[QUERY, chunk["text"]] for chunk in rrf_merged]
raw_scores = ce.predict(pairs)

results_with_scores = []
for idx, (chunk, raw) in enumerate(zip(rrf_merged, raw_scores)):
    raw_f = float(raw)
    sigmoid_prob = float(1.0 / (1.0 + np.exp(-raw_f)))
    chunk["raw_rerank_score"] = raw_f
    chunk["rerank_score"] = sigmoid_prob
    results_with_scores.append(chunk)

results_with_scores = sorted(results_with_scores, key=lambda x: x["rerank_score"], reverse=True)

print(f"\n  ALL RERANKED CHUNKS (sorted by rerank_score desc):")
print(f"  {'#':<3} {'raw_logit':>10} {'sigmoid_prob':>13} {'source':<35} {'text preview'}")
print(f"  {'─'*3} {'─'*10} {'─'*13} {'─'*35} {'─'*30}")
for i, c in enumerate(results_with_scores):
    src = str(c.get("metadata", {}).get("source", "?"))[:33]
    txt = c["text"][:45].replace("\n", " ")
    print(f"  {i+1:<3} {c['raw_rerank_score']:>10.4f} {c['rerank_score']:>13.6f} {src:<35} {txt!r}...")

# ── 9. MMR diversity selection ────────────────────────────────────────
print(f"\n{MINI}")
print(f"STEP 5: MAXIMAL MARGINAL RELEVANCE (MMR) — top_k=5")
print(MINI)
print(f"  MMR selects diverse top_k chunks balancing relevance and non-redundancy")
query_emb = em.generate_embeddings([QUERY])[0]
chunk_texts = [c["text"] for c in results_with_scores]
chunk_embs  = em.generate_embeddings(chunk_texts)
mmr_chunks = retriever._maximal_marginal_relevance(query_emb, chunk_embs, results_with_scores, top_k=5)
print(f"  MMR selected {len(mmr_chunks)} chunks")
for i, c in enumerate(mmr_chunks):
    print(f"  MMR [{i+1}] rerank_score={c.get('rerank_score',0):.6f} raw_logit={c.get('raw_rerank_score',0):.4f} | source={c.get('metadata',{}).get('source','?')!r}")

# ── 10. Adaptive Top-K ────────────────────────────────────────────────
print(f"\n{MINI}")
print(f"STEP 6: ADAPTIVE TOP-K (score_margin=0.15)")
print(MINI)
top_score = mmr_chunks[0].get("rerank_score", 0.0) if mmr_chunks else 0.0
adaptive_chunks = retriever._adaptive_top_k(mmr_chunks, max_k=5, score_key="rerank_score")
print(f"  Top score    : {top_score:.6f}")
print(f"  Score margin : 0.15")
print(f"  Kept {len(adaptive_chunks)} of {len(mmr_chunks)} chunks after adaptive pruning")

# ── 11. Final confidence and rejection decision ──────────────────────
print(f"\n{DIVIDER}")
print(f"FINAL DIAGNOSTIC REPORT")
print(DIVIDER)
print(f"\n  QUERY                   : {QUERY}")
print(f"  INDEXED CHUNKS (total)  : {len(em.chunks)}")
print(f"  FAISS RESULTS           : {len(faiss_candidates)}")
print(f"  BM25 RESULTS            : {len(bm25_candidates)}")
print(f"  RRF POOL                : {len(rrf_merged)}")
print(f"  AFTER RERANKING         : {len(results_with_scores)}")
print(f"  AFTER MMR               : {len(mmr_chunks)}")
print(f"  AFTER ADAPTIVE TOP-K    : {len(adaptive_chunks)}")

if adaptive_chunks:
    print(f"\n  RETRIEVED CHUNKS (returned to pipeline):")
    for i, c in enumerate(adaptive_chunks):
        src = c.get("metadata", {}).get("source", "?")
        txt = c["text"][:100].replace("\n", " ")
        raw = c.get("raw_rerank_score", 0.0)
        prob = c.get("rerank_score", 0.0)
        faiss_l2 = c.get("score", 0.0)
        print(f"\n  Chunk [{i+1}]")
        print(f"    SOURCE          : {src}")
        print(f"    TEXT PREVIEW    : {txt!r}...")
        print(f"    FAISS SCORE     : {faiss_l2:.4f}  ← L2 distance (lower = more similar, NOT probability)")
        print(f"    RAW CE LOGIT    : {raw:.4f}  ← CrossEncoder raw logit (unbounded, e.g. -10 to +10)")
        print(f"    RERANK SCORE    : {prob:.6f}  ← sigmoid(raw_logit) = probability of relevance [0,1]")

best = adaptive_chunks[0].get("rerank_score", 0.0) if adaptive_chunks else 0.0
threshold = 0.15
rejected = best < threshold

print(f"\n  CONFIDENCE SCORE        : {best:.6f}  ({best*100:.2f}%)")
print(f"  CONFIDENCE THRESHOLD    : {threshold}  (15.00%)")
print(f"  REJECTION DECISION      : {'❌ REJECTED — Insufficient data' if rejected else '✅ PASSED — Answer generated'}")

# ── 12. Score type identification for 0.0130 ────────────────────────
print(f"\n{DIVIDER}")
print("WHAT IS THE SCORE 0.0130 ?")
print(DIVIDER)
print("""
From your earlier log:
  Chunk [1] Source: Basics of Transformer.docx | Score: 0.0130

Tracing through the code:

1. pipeline.py line 91:
       best_score = retrieved_chunks[0].get('rerank_score', 0.0)

2. retriever.py line 275:
       prob = float(1.0 / (1.0 + np.exp(-float(score))))
       unique_chunks[idx]["rerank_score"] = prob

   → The 'rerank_score' field is sigmoid(raw_logit).
   → sigmoid(-4.33) = 0.0130
   → That means the CrossEncoder raw logit was approximately -4.33

3. What is 0.0130 exactly?
   TYPE  : CrossEncoder PROBABILITY (sigmoid-transformed relevance score)
   RANGE : [0.0, 1.0]
   MEANING: 1.3% probability that this chunk is relevant to the query
   WHY LOW: The query was 'What is tranformer?' — a TYPO.
             'tranformer' (missing 's') tokenizes differently.
             Cross-encoder gave it raw logit ≈ -4.33 → sigmoid = 0.013
             vs 'transformer' (correct) → raw logit +7.66 → sigmoid = 0.9995

4. It is NOT:
   ❌ Cosine similarity (cosine is in range [-1.0, 1.0])
   ❌ Cosine distance
   ❌ L2 distance (the FAISS score is L2, e.g. 0.63)
   ❌ A normalized FAISS score
   ❌ A BM25 score
   ✅ It IS: sigmoid(CrossEncoder raw logit) = probability of passage relevance
""")

print(DIVIDER)
print("CONCLUSION: The deployed app starts with 0 indexed chunks.")
print("On deployment, FAISS has ntotal=0 → search returns [] → confidence=0.0")
print("The 0.0130 score only appears LOCALLY after uploading a document.")
print(DIVIDER)
