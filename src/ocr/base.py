from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseOCREngine(ABC):
    @abstractmethod
    def extract_text_with_metadata(self, image_data: bytes) -> List[Dict[str, Any]]:
        """
        Perform OCR on raw image bytes.
        
        Args:
            image_data: Raw bytes representing the image (PNG, JPEG, etc.).
            
        Returns:
            A list of dictionary objects for each detected text region:
            [
                {
                    "text": "Detected text line",
                    "coordinates": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]],  # Polygon points
                    "confidence": 0.95                                         # Model confidence score
                }
            ]
        """
        pass
