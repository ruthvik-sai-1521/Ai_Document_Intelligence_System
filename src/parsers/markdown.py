from datetime import datetime
from typing import List, Dict, Any
from src.parsers.base import BaseParser
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class MarkdownParser(BaseParser):
    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Markdown (.md) bytes as UTF-8."""
        source_name = metadata.get("source", "Unknown Markdown File")
        timestamp = datetime.now().isoformat()
        try:
            text = raw_data.decode("utf-8", errors="replace")
            logger.info(f"Successfully read Markdown file {source_name}")
            return [{
                "text": text,
                "metadata": {
                    "source": source_name,
                    "timestamp": timestamp,
                    "page_number": 1
                }
            }]
        except Exception as e:
            logger.error(f"Error reading Markdown file {source_name}: {e}")
            return []
