import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingManager:
    """
    A comprehensive reusable module for generating embeddings and managing the FAISS vector store.
    """
    def __init__(self, model_name: str = "all-mpnet-base-v2", index_path: Optional[Path] = None, chunks_path: Optional[Path] = None):
        logger.info(f"Initializing EmbeddingManager with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        # Automatically get dimension from the model
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.index_path = index_path
        self.chunks_path = chunks_path
        
        self.index = None
        self.chunks = []
        self._embedding_cache = {}
        
        # Load existing index if paths are provided and exist
        if self.index_path and self.chunks_path and self.index_path.exists() and self.chunks_path.exists():
            self.load_index()
        else:
            self._initialize_empty_index()

    def _initialize_empty_index(self):
        """Initializes an empty FAISS index using L2 distance."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        logger.info(f"Initialized new empty FAISS index with dimension {self.dimension}.")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts, utilizing an in-memory cache."""
        if not texts:
            return np.array([])
            
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []
        
        # Check cache
        for i, text in enumerate(texts):
            if text in self._embedding_cache:
                embeddings.append(self._embedding_cache[text])
            else:
                embeddings.append(None) # Placeholder
                texts_to_compute.append(text)
                indices_to_compute.append(i)
                
        if texts_to_compute:
            logger.info(f"Generating new embeddings for {len(texts_to_compute)} texts (Cache hits: {len(texts) - len(texts_to_compute)})...")
            computed_embeddings = self.model.encode(texts_to_compute, show_progress_bar=False)
            
            for idx, emb, text in zip(indices_to_compute, computed_embeddings, texts_to_compute):
                emb_array = np.array(emb).astype('float32')
                embeddings[idx] = emb_array
                self._embedding_cache[text] = emb_array
                
        return np.array(embeddings).astype('float32')

    def add_chunks(self, chunks: List[Dict[str, Any]], save: bool = True):
        """
        Convert chunks to embeddings and store them in the FAISS index.
        Each chunk must have a 'text' key.
        """
        if not chunks:
            logger.warning("No chunks provided to add.")
            return

        texts = [chunk['text'] for chunk in chunks]
        embeddings_np = self.generate_embeddings(texts)
        
        self.index.add(embeddings_np)
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total chunks: {len(self.chunks)}.")
        
        if save and self.index_path and self.chunks_path:
            self.save_index()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Convert a query to an embedding and search the FAISS index.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty. Cannot search.")
            return []

        query_embedding = self.generate_embeddings([query])
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(distances[0][i])
                results.append(chunk)

        logger.info(f"Found {len(results)} FAISS results for query.")
        return results

    def save_index(self, index_path: Optional[Path] = None, chunks_path: Optional[Path] = None):
        """Save the FAISS index and chunk metadata to disk."""
        target_index = index_path or self.index_path
        target_chunks = chunks_path or self.chunks_path
        
        if not target_index or not target_chunks:
            logger.error("Cannot save: Index path or chunks path not specified.")
            return

        # Ensure directories exist
        target_index.parent.mkdir(parents=True, exist_ok=True)
        target_chunks.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(target_index))
        with open(target_chunks, "wb") as f:
            pickle.dump(self.chunks, f)
            
        logger.info(f"Successfully saved FAISS index and chunks.")

    def load_index(self, index_path: Optional[Path] = None, chunks_path: Optional[Path] = None):
        """Load the FAISS index and chunk metadata from disk."""
        target_index = index_path or self.index_path
        target_chunks = chunks_path or self.chunks_path

        if target_index and target_index.exists() and target_chunks and target_chunks.exists():
            logger.info("Loading existing FAISS index and chunks from disk (Memory-Mapped)...")
            self.index = faiss.read_index(str(target_index), faiss.IO_FLAG_MMAP)
            with open(target_chunks, "rb") as f:
                self.chunks = pickle.load(f)
            logger.info(f"Loaded successfully. Total chunks in index: {len(self.chunks)}")
        else:
            logger.warning("Index or chunks file not found. Cannot load.")
