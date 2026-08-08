from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseConnector(ABC):
    @abstractmethod
    def fetch_documents(self) -> List[Dict[str, Any]]:
        """
        Fetch documents from the source.
        Returns a list of dictionaries with document contents and metadata.
        """
        pass
