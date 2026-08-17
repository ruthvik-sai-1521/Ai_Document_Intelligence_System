import io
try:
    import easyocr
except ImportError:
    easyocr = None
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
try:
    from src.ocr.base import BaseOCREngine
    from src.core.logger import setup_logger
except ImportError:
    from ocr.base import BaseOCREngine
    from core.logger import setup_logger

logger = setup_logger(__name__)

class EasyOCREngine(BaseOCREngine):
    _readers_cache = {}

    def __init__(self, languages: Optional[List[str]] = None):
        if languages is None:
            languages = ['en']
        cache_key = tuple(sorted(languages))
        if easyocr is None:
            logger.warning("easyocr module is not installed. OCR capability disabled.")
            self.reader = None
        elif cache_key in EasyOCREngine._readers_cache:
            logger.debug(f"Reusing cached EasyOCREngine for languages: {languages}")
            self.reader = EasyOCREngine._readers_cache[cache_key]
        else:
            logger.info(f"Initializing EasyOCREngine for languages: {languages}...")
            # gpu=False is set by default to prevent CUDA compatibility warnings on standard CPU machines
            self.reader = easyocr.Reader(languages, gpu=False)
            EasyOCREngine._readers_cache[cache_key] = self.reader
        
    def extract_text_with_metadata(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Run OCR on raw image bytes and return text blocks with coordinates and confidences."""
        results = []
        try:
            image = Image.open(io.BytesIO(image_data))
            img_np = np.array(image)
            
            # readtext returns: [([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], text, confidence), ...]
            ocr_results = self.reader.readtext(img_np)
            
            for bbox, text, prob in ocr_results:
                coordinates = [[float(pt[0]), float(pt[1])] for pt in bbox]
                results.append({
                    "text": text,
                    "coordinates": coordinates,
                    "confidence": float(prob)
                })
        except Exception as e:
            logger.error(f"EasyOCR extraction error: {e}")
        return results
