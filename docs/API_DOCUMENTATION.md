# 📖 Python API Specifications & Core Reference

This document provides developer reference documentation for the core classes, methods, signatures, and data contracts in the **DocuMind AI** codebase.

---

## 1. Authentication & Security (`src/core/auth.py`)

### `hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]`
Generates a SHA-256 password hash using a 16-byte random salt.
- **Parameters**: `password` (raw plaintext password), `salt` (optional 16-byte hex salt).
- **Returns**: `(hashed_password, salt)` tuple.

### `register_user(username: str, password: str, role: str = "user") -> Tuple[bool, str]`
Registers a new user in the SQLite user database.
- **Parameters**: `username`, `password`, `role` (`"user"` or `"admin"`).
- **Returns**: `(success_flag, status_message)` tuple.

### `authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]`
Validates user credentials against the SQLite database.
- **Returns**: Dict containing `{"user_id", "username", "role"}` if valid, else `None`.

### `create_access_token(user_data: Dict[str, Any], expires_delta_hours: int = 24) -> str`
Encodes a JWT session token containing user payload claims.
- **Returns**: Encoded JWT string token.

### `decode_access_token(token: str) -> Optional[Dict[str, Any]]`
Decodes and validates a JWT token signature and expiration (`exp`).
- **Returns**: User claim dictionary if valid, else `None`.

---

## 2. Vector & Lexical Engines

### `EmbeddingManager` (`src/core/embedding_manager.py`)
```python
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: Path = None, chunks_path: Path = None)
```
- **`generate_embeddings(texts: List[str]) -> np.ndarray`**: Generates normalized $d=384$ vector embeddings for input text snippets.
- **`add_chunks(chunks: List[Dict[str, Any]])`**: Encodes text chunks and inserts them into the FAISS vector index and memory cache.
- **`search(query: str, top_k: int = 5, user_id: str = None, filters: dict = None) -> List[Dict[str, Any]]`**: Performs cosine similarity vector search filtered by `user_id` or metadata tags.
- **`remove_document(filename: str)`**: Removes all vectors belonging to `filename` from the active FAISS index.

### `KeywordSearch` (`src/retrieval/keyword_search.py`)
```python
class KeywordSearch:
    def __init__(self, index_path: Path = None)
```
- **`add_chunks(chunks: List[Dict[str, Any]])`**: Tokenizes text chunks using Okapi BM25 and updates index state.
- **`search(query: str, top_k: int = 5, user_id: str = None, filters: dict = None) -> List[Dict[str, Any]]`**: Executes BM25 lexical keyword search filtered by `user_id`.

---

## 3. Hybrid Retriever & Re-Ranker (`src/retrieval/retriever.py`)

### `HybridRetriever`
```python
class HybridRetriever:
    def __init__(self, embedding_manager: EmbeddingManager, keyword_search: KeywordSearch)
```

#### `retrieve(query: str, top_k: int = 5, user_id: str = None, filters: dict = None) -> List[Dict[str, Any]]`
Executes multi-stage hybrid retrieval:
1. Candidate retrieval from FAISS (vector) and BM25 (keyword).
2. **Reciprocal Rank Fusion (RRF)**: Merges rank positions using $RRF\_Score = \sum \frac{1}{60 + r}$.
3. **CrossEncoder Re-ranking**: Evaluates candidates using `sentence-transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`.
4. **Maximal Marginal Relevance (MMR)**: Applies $\lambda = 0.7$ diversification to prune redundant chunks.
5. **Context Compression**: Sentence-level semantic filtering.
6. **Adaptive Top-K**: Prunes candidates if re-ranking score drops sharply.

- **Returns**: Ordered list of relevant candidate chunk dictionaries.

---

## 4. LLM Generation & Pipeline (`src/core/pipeline.py`)

### `RAGPipeline`
```python
class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLMGenerator, confidence_threshold: float = 0.15)
```

#### `run(query: str, user_id: str = None, session_id: str = "default") -> Tuple[str, Dict[str, Any]]`
Executes end-to-end RAG pipeline:
1. Loads session dialog history (`limit=10`).
2. Applies token/word sliding window optimization (pruning older turns over 600 words).
3. Executes LLM query rewriting for follow-up questions.
4. Executes Hybrid Search & Re-ranking.
5. Checks confidence score against `confidence_threshold`.
6. Generates final answer with citations.
7. Measures high-resolution timing breakdown (`embedding_time_ms`, `llm_time`, `retrieval_time`, `latency`).

- **Returns**: `(answer_text, metadata_dict)` tuple.

---

## 5. Quantitative Evaluator (`src/evaluation/evaluator.py`)

### `RAGEvaluator`
```python
class RAGEvaluator:
    def __init__(self, pipeline: RAGPipeline, results_dir: str = "logs")
```
- **`evaluate_single_query(query: str, expected_keywords: List[str] = None) -> Dict[str, Any]`**: Calculates all 10 core metrics for a single query.
- **`run_benchmark() -> Dict[str, Any]`**: Executes benchmark evaluation suite across test cases and returns mean metrics summary.
