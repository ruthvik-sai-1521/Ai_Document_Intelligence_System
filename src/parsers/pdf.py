import io
import PyPDF2
from typing import List, Dict, Any
from src.parsers.base import BaseParser
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class PDFParser(BaseParser):
    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse raw PDF byte stream and extract text from each page."""
        pages = []
        source_name = metadata.get("source", "Unknown PDF")
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        try:
            file_like = io.BytesIO(raw_data)
            reader = PyPDF2.PdfReader(file_like)
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    pages.append({
                        "text": extracted,
                        "metadata": {
                            "source": source_name,
                            "timestamp": timestamp,
                            "page_number": i + 1
                        }
                    })
            logger.info(f"Successfully extracted {len(pages)} pages from {source_name}")
        except Exception as e:
            logger.error(f"Error reading PDF {source_name}: {e}")
        return pages
