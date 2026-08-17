from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseEmbedding(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of document strings.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            A 2D numpy array of shape (num_texts, dimension) containing float32 embeddings.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        
        Args:
            text: Query string to embed.
            
        Returns:
            A 1D numpy array containing float32 embedding.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return dimension of the embedding model."""
        pass
