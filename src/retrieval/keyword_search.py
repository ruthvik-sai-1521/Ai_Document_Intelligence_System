from rank_bm25 import BM25Okapi
import pickle
from typing import List, Dict, Any
from pathlib import Path
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class KeywordSearch:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.bm25 = None
        self.chunks = []
        self._load()

    def _load(self):
        """Load BM25 index and chunks if they exist."""
        if self.index_path.exists():
            logger.info("Loading BM25 index from disk.")
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data.get("bm25")
                self.chunks = data.get("chunks", [])
            logger.info("Loaded successfully.")
        else:
            logger.info("No existing BM25 index found.")

    def save(self):
        """Save BM25 index and chunks to disk."""
        if self.bm25 is not None:
            with open(self.index_path, "wb") as f:
                pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
            logger.info("Saved BM25 index to disk.")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        return text.lower().split()

    def add_chunks(self, new_chunks: List[Dict[str, Any]]):
        """Add new chunks to the global corpus and rebuild the BM25 index."""
        if not new_chunks:
            logger.warning("No chunks provided to add_chunks.")
            return

        logger.info(f"Adding {len(new_chunks)} new chunks to BM25 index...")
        self.chunks.extend(new_chunks)
        
        # BM25 requires the entire corpus for accurate inverse document frequencies
        tokenized_corpus = [self._tokenize(chunk['text']) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the BM25 index for the most relevant chunks."""
        if self.bm25 is None or not self.chunks:
            logger.warning("BM25 index is empty during search.")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_n_indices:
            if scores[idx] > 0:
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(scores[idx])
                results.append(chunk)

        return results
