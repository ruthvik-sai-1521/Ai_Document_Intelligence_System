from typing import List, Dict, Any
from src.parsers.base import BaseParser
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class TXTParser(BaseParser):
    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse raw text bytes and return standard single page dict."""
        source_name = metadata.get("source", "Unknown Text File")
        try:
            text = raw_data.decode("utf-8", errors="replace")
            logger.info(f"Successfully read text file {source_name}")
            return [{"page_num": 1, "text": text}]
        except Exception as e:
            logger.error(f"Error reading text file {source_name}: {e}")
            return []
