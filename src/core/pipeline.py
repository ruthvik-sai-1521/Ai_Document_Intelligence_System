from typing import Dict, Any, Tuple
from src.retrieval.retriever import HybridRetriever
from src.llm.generator import LLMGenerator
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLMGenerator, confidence_threshold: float = 0.15):
        """
        Initialize the RAG Pipeline.
        confidence_threshold: Minimum cosine similarity score required to answer.
        """
        self.retriever = retriever
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self._query_cache = {}

    def run(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Executes the full RAG pipeline:
        1. Accepts query
        2. Performs hybrid search
        3. Re-ranks results
        4. Computes confidence score (rejects if low)
        5. Builds context and generates Gemini answer
        """
        query_key = query.strip().lower()
        if query_key in self._query_cache:
            logger.info(f"--- Cache hit for query: '{query}' ---")
            return self._query_cache[query_key]
            
        logger.info(f"--- Running RAG pipeline for query: '{query}' ---")

        try:
            # Steps 1, 2 & 3: Hybrid search and Re-ranking
            retrieved_chunks = self.retriever.retrieve(query, top_k=5)

            if not retrieved_chunks:
                logger.warning("No documents retrieved.")
                return "Insufficient data", {"confidence": 0.0, "sources": []}

            # Log retrieved chunks
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query.")
            for i, chunk in enumerate(retrieved_chunks):
                source = chunk.get('metadata', {}).get('source', 'Unknown')
                score = chunk.get('rerank_score', 0.0)
                # Sanitize snippet to ASCII to prevent Windows cp1252 console crash
                snippet = chunk['text'][:100].encode('ascii', errors='replace').decode('ascii') + "..."
                logger.info(f"Chunk [{i+1}] Source: {source} | Score: {score:.4f} | Text: {snippet}")

            # Calculate Confidence Score using the top rerank score (Cosine Similarity)
            best_score = retrieved_chunks[0].get('rerank_score', 0.0)
            logger.info(f"Query confidence score: {best_score:.4f} (Threshold: {self.confidence_threshold})")

            # Check for low confidence
            if best_score < self.confidence_threshold:
                logger.warning(f"Low confidence ({best_score:.4f} < {self.confidence_threshold}). Rejecting query.")
                return "Insufficient data", {
                    "confidence": best_score,
                    "sources": retrieved_chunks
                }

            # The llm.generate_answer method inherently builds the formatted context from the chunks
            answer = self.llm.generate_answer(query, retrieved_chunks)
            
            logger.info(f"Successfully generated final response. Preview: {answer[:150].replace(chr(10), ' ')}...")
            result = (answer, {
                "confidence": best_score,
                "sources": retrieved_chunks
            })
            
            self._query_cache[query_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error during RAG pipeline execution: {str(e)}", exc_info=True)
            return f"An error occurred: {str(e)}", {"confidence": 0.0, "sources": []}
