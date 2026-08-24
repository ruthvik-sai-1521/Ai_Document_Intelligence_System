import json
from typing import List, Dict, Any
from src.parsers.base import BaseParser
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class StructuredDataParser(BaseParser):
    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parses structured database row payloads and converts them into semantic text pages.
        Payload format: {
            "table_name": str,
            "primary_key_col": str,
            "columns": List[str],
            "rows": List[List[Any]]
        }
        """
        source_name = metadata.get("source", "Unknown Table")
        
        try:
            payload = json.loads(raw_data.decode("utf-8", errors="replace"))
        except Exception as e:
            logger.error(f"Failed to parse JSON payload for structured data: {e}")
            return [{"text": raw_data.decode("utf-8", errors="replace"), "metadata": metadata}]
            
        if not isinstance(payload, dict) or "rows" not in payload:
            # Fallback for standard JSON configuration/data files
            pretty_json = json.dumps(payload, indent=2)
            return [{"text": pretty_json, "metadata": metadata}]
            
        table_name = payload.get("table_name", "Unknown Table")
        pk_col = payload.get("primary_key_col")
        columns = payload.get("columns", [])
        rows = payload.get("rows", [])
        
        pages = []
        for i, row in enumerate(rows):
            # Extract primary key value if columns match
            pk_val = None
            if pk_col and pk_col in columns:
                try:
                    pk_idx = columns.index(pk_col)
                    pk_val = str(row[pk_idx])
                except Exception:
                    pk_val = f"Index_{i+1}"
            else:
                pk_val = f"Index_{i+1}"
                
            # Build semantic representation
            text_lines = [
                f"Table: {table_name}",
                "Column Headers: " + " | ".join(columns),
                f"Row Data (Primary Key {pk_col or 'Index'} = {pk_val}):"
            ]
            for col_name, val in zip(columns, row):
                text_lines.append(f"- {col_name}: {val if val is not None else 'NULL'}")
                
            row_text = "\n".join(text_lines)
            
            # Preserve structural metadata
            row_metadata = metadata.copy()
            row_metadata.update({
                "source_type": "structured_row",
                "table_name": table_name,
                "primary_key_column": pk_col or "Index",
                "primary_key_value": pk_val,
                "columns": columns,
                "row_number": i + 1
            })
            
            pages.append({
                "text": row_text,
                "metadata": row_metadata
            })
            
        logger.info(f"Successfully chunked {len(pages)} rows from table {table_name}")
        return pages
