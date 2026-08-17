from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.embeddings.base import BaseEmbedding
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing HuggingFaceEmbedding with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate float32 embeddings for a list of texts."""
        if not texts:
            return np.array([], dtype='float32')
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings).astype('float32')

    def embed_query(self, text: str) -> np.ndarray:
        """Generate float32 embedding for a query string."""
        embedding = self.model.encode([text], show_progress_bar=False)[0]
        return np.array(embedding).astype('float32')
        
    def get_embedding_dimension(self) -> int:
        """Helper to get dimension size from model."""
        dim = None
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        if dim is None:
            dim = self.model.get_sentence_embedding_dimension()
        return int(dim or 384)
