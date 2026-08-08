import io
import csv
from datetime import datetime
from typing import List, Dict, Any
from src.parsers.base import BaseParser
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class CsvParser(BaseParser):
    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse CSV bytes, grouping rows in blocks to keep RAG chunks aligned."""
        source_name = metadata.get("source", "Unknown CSV File")
        timestamp = datetime.now().isoformat()
        pages = []
        try:
            text_stream = io.StringIO(raw_data.decode("utf-8", errors="replace"))
            reader = csv.reader(text_stream)
            
            current_rows = []
            start_row_num = 1
            batch_size = 50  # Process in blocks of 50 rows
            
            for row_idx, row in enumerate(reader):
                row_str = " | ".join([cell.strip() for cell in row])
                if row_str.strip():
                    current_rows.append(row_str)
                
                if (row_idx + 1) % batch_size == 0:
                    if current_rows:
                        pages.append({
                            "text": "\n".join(current_rows),
                            "metadata": {
                                "source": source_name,
                                "timestamp": timestamp,
                                "row_number": start_row_num
                            }
                        })
                        current_rows = []
                        start_row_num = row_idx + 2
            
            if current_rows:
                pages.append({
                    "text": "\n".join(current_rows),
                    "metadata": {
                        "source": source_name,
                        "timestamp": timestamp,
                        "row_number": start_row_num
                    }
                })
                
            logger.info(f"Successfully extracted {len(pages)} blocks from CSV {source_name}")
        except Exception as e:
            logger.error(f"Error parsing CSV {source_name}: {e}")
        return pages
