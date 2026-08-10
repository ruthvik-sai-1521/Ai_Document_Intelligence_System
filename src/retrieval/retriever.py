from typing import List, Dict, Any, Optional
import numpy as np
import re
from retrieval.keyword_search import KeywordSearch
from core.embedding_manager import EmbeddingManager
from core.logger import setup_logger

logger = setup_logger(__name__)

class HybridRetriever:
    def __init__(self, embedding_manager: EmbeddingManager, keyword_search: KeywordSearch):
        self.embedding_manager = embedding_manager
        self.keyword_search = keyword_search

    @property
    def cross_encoder(self):
        if not hasattr(self, "_cross_encoder") or self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            logger.info("Initializing CrossEncoder re-ranker model: cross-encoder/ms-marco-MiniLM-L-6-v2...")
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._cross_encoder

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
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

    def _maximal_marginal_relevance(
        self,
        query_emb: np.ndarray,
        chunk_embs: np.ndarray,
        chunks: List[Dict[str, Any]],
        top_k: int,
        lambda_val: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks chunks using Maximal Marginal Relevance (MMR) to balance relevance and diversity.
        """
        if not chunks or top_k <= 0:
            return []
            
        n = len(chunks)
        selected_indices = []
        remaining_indices = list(range(n))
        
        # Cosine similarity with query for relevance
        query_sims = np.array([self._cosine_similarity(query_emb, chunk_embs[i]) for i in range(n)])
        
        # Similarity matrix between chunks for diversity
        doc_sims = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = self._cosine_similarity(chunk_embs[i], chunk_embs[j])
                doc_sims[i, j] = sim
                doc_sims[j, i] = sim
                
        while len(selected_indices) < min(top_k, n):
            best_mmr = -float('inf')
            best_idx = -1
            
            for idx in remaining_indices:
                relevance = query_sims[idx]
                if selected_indices:
                    redundancy = max(doc_sims[idx, s_idx] for s_idx in selected_indices)
                else:
                    redundancy = 0.0
                    
                mmr_score = lambda_val * relevance - (1 - lambda_val) * redundancy
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
                    
            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
        return [chunks[idx] for idx in selected_indices]

    def _compress_context(
        self,
        chunk: Dict[str, Any],
        query_emb: np.ndarray,
        threshold: float = 0.35,
        min_sentences: int = 1
    ) -> Dict[str, Any]:
        """
        Compresses context dynamically by retaining only sentences with high semantic similarity to query.
        """
        text = chunk.get("text", "")
        if not text:
            return chunk
            
        # Basic sentence boundary splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= min_sentences:
            return chunk
            
        try:
            sentence_embs = self.embedding_manager.generate_embeddings(sentences)
        except Exception as e:
            logger.error(f"Failed to generate sentence embeddings for compression: {e}")
            return chunk
            
        scored_sentences = []
        for i, s_emb in enumerate(sentence_embs):
            sim = self._cosine_similarity(query_emb, s_emb)
            scored_sentences.append((sim, i, sentences[i]))
            
        retained = []
        for sim, idx, sentence in scored_sentences:
            if sim >= threshold:
                retained.append((idx, sentence))
                
        # Keep at least min_sentences if none pass threshold
        if len(retained) < min_sentences:
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            for i in range(min(min_sentences, len(scored_sentences))):
                sim, idx, sentence = scored_sentences[i]
                retained.append((idx, sentence))
                
        retained.sort(key=lambda x: x[0])
        compressed_text = " ".join([item[1] for item in retained])
        
        compressed_chunk = chunk.copy()
        compressed_chunk["text"] = compressed_text
        compressed_chunk["original_text"] = text
        compressed_chunk["compressed"] = True
        return compressed_chunk

    def _adaptive_top_k(
        self,
        chunks: List[Dict[str, Any]],
        max_k: int,
        score_margin: float = 0.15,
        score_key: str = "rerank_score"
    ) -> List[Dict[str, Any]]:
        """
        Dynamically adjusts the number of returned chunks based on score drop-off margin.
        """
        if not chunks:
            return []
            
        top_score = chunks[0].get(score_key, 0.0)
        retained_chunks = [chunks[0]]
        
        for chunk in chunks[1:max_k]:
            score = chunk.get(score_key, 0.0)
            if score_key == "rerank_score":
                # For cross-encoder logit scores (e.g. logits scale), a margin of 3.0 represents a significant drop-off
                if top_score - score > 3.0:
                    break
            else:
                if top_score - score > score_margin:
                    break
            retained_chunks.append(chunk)
            
        return retained_chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 15,
        user_id: str = None,
        filters: Dict[str, Any] = None,
        use_reranker: bool = True,
        use_mmr: bool = True,
        use_compression: bool = True,
        use_adaptive_k: bool = True,
        lambda_val: float = 0.5,
        compression_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Upgraded Hybrid Retrieval pipeline with CrossEncoder, MMR, Metadata Filtering, Context Compression, and Adaptive Top-K.
        """
        logger.info(f"Running Upgraded Retrieval for query: {query}")
        
        # Standardize filters dictionary for Metadata Filtering
        if filters is None:
            filters = {}
        if user_id:
            filters["user_id"] = user_id
            
        # 1. Retrieve candidates from Vector search (FAISS)
        # Note: We fetch retrieve_k * 3 items to allow ample candidates for metadata filtering
        faiss_candidates = self.embedding_manager.search(query, top_k=retrieve_k * 3, user_id=None)
        
        # 2. Retrieve candidates from Lexical search (BM25)
        bm25_candidates = self.keyword_search.search(query, top_k=retrieve_k * 3, user_id=None)
        
        # 3. Apply Metadata Filtering to both candidate streams
        filtered_faiss = []
        for c in faiss_candidates:
            meta = c.get("metadata", {})
            match = True
            for k, v in filters.items():
                if meta.get(k) != v:
                    match = False
                    break
            if match:
                filtered_faiss.append(c)
                if len(filtered_faiss) >= retrieve_k:
                    break
                    
        filtered_bm25 = []
        for c in bm25_candidates:
            meta = c.get("metadata", {})
            match = True
            for k, v in filters.items():
                if meta.get(k) != v:
                    match = False
                    break
            if match:
                filtered_bm25.append(c)
                if len(filtered_bm25) >= retrieve_k:
                    break

        # 4. Hybrid Search Merge using Reciprocal Rank Fusion (RRF)
        unique_chunks = self._reciprocal_rank_fusion(filtered_faiss, filtered_bm25)
        if not unique_chunks:
            return []
            
        # Limit to unique merged pool size before re-ranking
        unique_chunks = unique_chunks[:retrieve_k]
        
        # 5. Cross-Encoder Re-ranking
        if use_reranker:
            try:
                pairs = [[query, chunk["text"]] for chunk in unique_chunks]
                scores = self.cross_encoder.predict(pairs)
                for idx, score in enumerate(scores):
                    unique_chunks[idx]["rerank_score"] = float(score)
                unique_chunks = sorted(unique_chunks, key=lambda x: x["rerank_score"], reverse=True)
            except Exception as e:
                logger.error(f"Cross-Encoder re-ranking failed, falling back to cosine similarity: {e}")
                use_reranker = False
                
        # Cosine Similarity backup if Cross-Encoder is disabled/failed
        query_emb = self.embedding_manager.generate_embeddings([query])[0]
        chunk_texts = [chunk["text"] for chunk in unique_chunks]
        chunk_embs = self.embedding_manager.generate_embeddings(chunk_texts)
        
        if not use_reranker:
            for idx, chunk in enumerate(unique_chunks):
                sim = self._cosine_similarity(query_emb, chunk_embs[idx])
                chunk["score"] = sim
                # Make sure rerank_score exists for compatibility
                chunk["rerank_score"] = sim
            unique_chunks = sorted(unique_chunks, key=lambda x: x["score"], reverse=True)
            
        # 6. Diversify chunks using MMR (Maximal Marginal Relevance)
        if use_mmr:
            selected_chunks = self._maximal_marginal_relevance(
                query_emb=query_emb,
                chunk_embs=chunk_embs,
                chunks=unique_chunks,
                top_k=top_k,
                lambda_val=lambda_val
            )
        else:
            selected_chunks = unique_chunks[:top_k]
            
        if not selected_chunks:
            return []
            
        # 7. Adaptive Top-K pruning
        if use_adaptive_k:
            selected_chunks = self._adaptive_top_k(
                chunks=selected_chunks,
                max_k=top_k,
                score_key="rerank_score" if use_reranker else "score"
            )
            
        # 8. Dynamic Context Compression
        if use_compression:
            compressed_chunks = []
            for chunk in selected_chunks:
                comp_chunk = self._compress_context(
                    chunk=chunk,
                    query_emb=query_emb,
                    threshold=compression_threshold
                )
                compressed_chunks.append(comp_chunk)
            selected_chunks = compressed_chunks
            
        return selected_chunks
