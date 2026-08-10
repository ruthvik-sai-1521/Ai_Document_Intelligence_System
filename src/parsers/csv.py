import io
import csv
from datetime import datetime
from typing import List, Dict, Any
from src.parsers.base import BaseParser
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class CsvParser(BaseParser):
    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse CSV bytes, converting rows into semantic chunks."""
        source_name = metadata.get("source", "Unknown CSV File")
        timestamp = datetime.now().isoformat()
        pages = []
        try:
            text_stream = io.StringIO(raw_data.decode("utf-8", errors="replace"))
            reader = list(csv.reader(text_stream))
            if not reader:
                return []
                
            # Retrieve header columns (first row)
            headers = [h.strip() for h in reader[0] if h.strip()]
            
            # Find a potential primary key column (default to first column)
            pk_col = None
            for h in headers:
                if "id" in h.lower() or "key" in h.lower() or "code" in h.lower():
                    pk_col = h
                    break
            if not pk_col and headers:
                pk_col = headers[0]
                
            for i, row in enumerate(reader[1:]):
                row_values = [cell.strip() for cell in row]
                if not any(row_values):
                    continue
                
                # Retrieve primary key value
                pk_val = None
                if pk_col and pk_col in headers:
                    try:
                        pk_idx = headers.index(pk_col)
                        pk_val = row_values[pk_idx] if pk_idx < len(row_values) else f"Row_{i+1}"
                    except Exception:
                        pk_val = f"Row_{i+1}"
                else:
                    pk_val = f"Row_{i+1}"
                    
                # Build semantic text representation
                text_lines = [
                    f"Table: {source_name}",
                    "Column Headers: " + " | ".join(headers),
                    f"Row Data (Primary Key {pk_col or 'Index'} = {pk_val}):"
                ]
                for col_name, val in zip(headers, row_values):
                    text_lines.append(f"- {col_name}: {val}")
                    
                row_text = "\n".join(text_lines)
                
                row_metadata = metadata.copy()
                row_metadata.update({
                    "source_type": "structured_row",
                    "table_name": source_name,
                    "primary_key_column": pk_col or "Index",
                    "primary_key_value": pk_val,
                    "columns": headers,
                    "row_number": i + 1,
                    "timestamp": timestamp
                })
                
                pages.append({
                    "text": row_text,
                    "metadata": row_metadata
                })
                
            logger.info(f"Successfully chunked {len(pages)} rows from CSV {source_name}")
        except Exception as e:
            logger.error(f"Error parsing CSV {source_name}: {e}")
        return pages
