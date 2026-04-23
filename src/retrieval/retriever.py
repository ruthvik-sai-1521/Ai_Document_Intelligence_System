from typing import List, Dict, Any
import numpy as np
from src.retrieval.keyword_search import KeywordSearch
from src.core.embedding_manager import EmbeddingManager
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class HybridRetriever:
    def __init__(self, embedding_manager: EmbeddingManager, keyword_search: KeywordSearch):
        self.embedding_manager = embedding_manager
        self.keyword_search = keyword_search

    def _reciprocal_rank_fusion(self, semantic_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """Merge results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        merged_chunks = {}

        for rank, res in enumerate(semantic_results):
            text = res['text']
            if text not in rrf_scores:
                rrf_scores[text] = 0.0
                merged_chunks[text] = res.copy()
            rrf_scores[text] += 1.0 / (k + rank + 1)

        for rank, res in enumerate(keyword_results):
            text = res['text']
            if text not in rrf_scores:
                rrf_scores[text] = 0.0
                merged_chunks[text] = res.copy()
            rrf_scores[text] += 1.0 / (k + rank + 1)

        for text, score in rrf_scores.items():
            merged_chunks[text]['rrf_score'] = score

        unique_chunks = list(merged_chunks.values())
        return sorted(unique_chunks, key=lambda x: x['rrf_score'], reverse=True)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def retrieve(self, query: str, top_k: int = 5, retrieve_k: int = 10) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving chunks for query: {query}")
        
        # 1. Retrieve top 10 chunks from Semantic Search (FAISS)
        semantic_results = self.embedding_manager.search(query, top_k=retrieve_k)

        # 2. Retrieve top 10 chunks from Keyword Search (BM25)
        keyword_results = self.keyword_search.search(query, top_k=retrieve_k)

        # 3. Merge Results
        unique_chunks = self._reciprocal_rank_fusion(semantic_results, keyword_results)
        logger.info(f"Found {len(unique_chunks)} unique chunks after merging.")
        
        if not unique_chunks:
            return []

        # 4. Re-rank using Cosine Similarity for High Efficiency
        # Generate the embedding for the query
        query_emb = self.embedding_manager.generate_embeddings([query])[0]
        
        # Generate embeddings for the retrieved chunks
        chunk_texts = [chunk['text'] for chunk in unique_chunks]
        chunk_embs = self.embedding_manager.generate_embeddings(chunk_texts)

        # Score and rank
        for i, chunk in enumerate(unique_chunks):
            sim = self._cosine_similarity(query_emb, chunk_embs[i])
            chunk['rerank_score'] = sim

        # Sort descending by Cosine Similarity score
        reranked_chunks = sorted(unique_chunks, key=lambda x: x['rerank_score'], reverse=True)

        # Select best 3-5 chunks (using top_k)
        return reranked_chunks[:top_k]
