from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseParser(ABC):
    @abstractmethod
    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse binary raw data into structured page/text blocks.
        
        Args:
            raw_data: Binary data stream of the document.
            metadata: Metadata containing file names, timestamps, etc.
            
        Returns:
            A list of page dictionaries, e.g.:
            [
                {"page_num": 1, "text": "Extracted text page 1..."},
                {"page_num": 2, "text": "Extracted text page 2..."}
            ]
        """
        pass
