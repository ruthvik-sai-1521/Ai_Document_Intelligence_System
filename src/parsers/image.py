from datetime import datetime
from typing import List, Dict, Any
from parsers.base import BaseParser
from ocr.easy_ocr import EasyOCREngine
from core.logger import setup_logger

logger = setup_logger(__name__)

class ImageParser(BaseParser):
    def __init__(self, ocr_engine=None):
        if ocr_engine is None:
            # Reusable OCR Engine instance
            from core.config import OCR_LANGUAGES
            self.ocr_engine = EasyOCREngine(languages=OCR_LANGUAGES)
        else:
            self.ocr_engine = ocr_engine
            
    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse raw image bytes using OCR."""
        source_name = metadata.get("source", "Unknown Image File")
        timestamp = datetime.now().isoformat()
        try:
            logger.info(f"Running OCR extraction on image {source_name}...")
            ocr_blocks = self.ocr_engine.extract_text_with_metadata(raw_data)
            
            # Combine all block text into a single continuous layout string
            extracted_text = " ".join([block["text"] for block in ocr_blocks])
            
            logger.info(f"Successfully ran OCR. Extracted {len(ocr_blocks)} text blocks from {source_name}")
            return [{
                "text": extracted_text,
                "metadata": {
                    "source": source_name,
                    "timestamp": timestamp,
                    "page_number": 1,
                    "is_ocr": True,
                    "ocr_metadata": ocr_blocks
                }
            }]
        except Exception as e:
            logger.error(f"Error parsing image {source_name}: {e}")
            return []
