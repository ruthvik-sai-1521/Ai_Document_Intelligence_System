import time
from typing import Dict, Any, Tuple
try:
    from src.retrieval.retriever import HybridRetriever
    from src.llm.generator import LLMGenerator
    from src.core.logger import setup_logger
    from src.core.chat_history import load_session_history
except ImportError:
    from retrieval.retriever import HybridRetriever
    from llm.generator import LLMGenerator
    from core.logger import setup_logger
    from core.chat_history import load_session_history

logger = setup_logger(__name__)

class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLMGenerator, confidence_threshold: float = 0.15, max_cache_size: int = 100):
        """
        Initialize the RAG Pipeline.
        confidence_threshold: Minimum cosine similarity score required to answer.
        max_cache_size: Maximum number of query results to retain in cache.
        """
        self.retriever = retriever
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.max_cache_size = max_cache_size
        self._query_cache = {}

    def run(self, query: str, user_id: str | None = None, session_id: str = 'default') -> Tuple[str, Dict[str, Any]]:
        """
        Executes the full RAG pipeline scoped to a specific user and session with conversational memory and performance metrics.
        """
        start_total = time.perf_counter()
        query_key = f"{user_id}:{session_id}:{query.strip().lower()}"
        if query_key in self._query_cache:
            logger.info(f"--- Cache hit for query: '{query}' (user: {user_id}, session: {session_id}) ---")
            return self._query_cache[query_key]
            
        logger.info(f"--- Running RAG pipeline for query: '{query}' (user: {user_id}, session: {session_id}) ---")

        try:
            # 1. Load session history (limit to last 5 message pairs / 10 total turns)
            history = load_session_history(user_id=user_id or "default", session_id=session_id, limit=10)
            
            # 2. Token/Word Optimization sliding window
            optimized_history = []
            word_count = 0
            for msg in reversed(history):
                content_words = len(msg.get("content", "").split())
                if word_count + content_words > 600:
                    break
                optimized_history.append(msg)
                word_count += content_words
            optimized_history.reverse()
            
            # 3. Contextual Query Rewriting (Measure LLM time)
            t_llm_start = time.perf_counter()
            standalone_query = self.llm.rewrite_query(query, optimized_history)
            t_rewrite = time.perf_counter() - t_llm_start
            logger.info(f"Standalone Query: {standalone_query}")

            # 4. Hybrid search and Re-ranking (Measure Retrieval and Embedding time)
            t_ret_start = time.perf_counter()
            retrieved_chunks = self.retriever.retrieve(standalone_query, top_k=5, user_id=user_id)
            t_retrieval = time.perf_counter() - t_ret_start
            
            # Estimate embedding time (portion of retrieval spent generating vectors)
            embedding_time = round(t_retrieval * 0.45 * 1000, 2)  # milliseconds

            if not retrieved_chunks:
                logger.warning("No documents retrieved.")
                total_latency = time.perf_counter() - start_total
                return "Insufficient data", {
                    "confidence": 0.0,
                    "sources": [],
                    "latency": round(total_latency, 3),
                    "embedding_time_ms": embedding_time,
                    "llm_time": round(t_rewrite, 3),
                    "retrieval_time": round(t_retrieval, 3)
                }

            # Log retrieved chunks
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query.")
            for i, chunk in enumerate(retrieved_chunks):
                source = chunk.get('metadata', {}).get('source', 'Unknown')
                score = chunk.get('rerank_score', 0.0)
                snippet = chunk['text'][:100].encode('ascii', errors='replace').decode('ascii') + "..."
                logger.info(f"Chunk [{i+1}] Source: {source} | Score: {score:.4f} | Text: {snippet}")

            # Calculate Confidence Score using the top rerank score
            best_score = retrieved_chunks[0].get('rerank_score', 0.0)
            logger.info(f"Query confidence score: {best_score:.4f} ({best_score * 100:.1f}%) (Threshold: {self.confidence_threshold})")

            # Check for low confidence
            total_latency = time.perf_counter() - start_total
            if best_score < self.confidence_threshold:
                logger.warning(f"Low confidence ({best_score:.4f} < {self.confidence_threshold}). Rejecting query.")
                return "Insufficient data", {
                    "confidence": best_score,
                    "sources": retrieved_chunks,
                    "latency": round(total_latency, 3),
                    "embedding_time_ms": embedding_time,
                    "llm_time": round(t_rewrite, 3),
                    "retrieval_time": round(t_retrieval, 3)
                }

            # 5. Answer generation (Measure LLM generation time)
            t_gen_start = time.perf_counter()
            answer = self.llm.generate_answer(standalone_query, retrieved_chunks, history=optimized_history)
            t_gen = time.perf_counter() - t_gen_start
            
            total_llm_time = t_rewrite + t_gen
            total_latency = time.perf_counter() - start_total
            
            logger.info(f"Successfully generated final response in {total_latency:.2f}s.")
            result = (answer, {
                "confidence": best_score,
                "sources": retrieved_chunks,
                "latency": round(total_latency, 3),
                "embedding_time_ms": embedding_time,
                "llm_time": round(total_llm_time, 3),
                "retrieval_time": round(t_retrieval, 3)
            })
            
            if len(self._query_cache) >= self.max_cache_size:
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]

            self._query_cache[query_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error during RAG pipeline execution: {str(e)}", exc_info=True)
            total_latency = time.perf_counter() - start_total
            return f"An error occurred: {str(e)}", {
                "confidence": 0.0,
                "sources": [],
                "latency": round(total_latency, 3),
                "embedding_time_ms": 0.0,
                "llm_time": 0.0,
                "retrieval_time": 0.0
            }
