import io
import openpyxl
from datetime import datetime
from typing import List, Dict, Any
from src.parsers.base import BaseParser
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class XlsxParser(BaseParser):
    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Excel workbook (.xlsx) bytes."""
        source_name = metadata.get("source", "Unknown Excel File")
        timestamp = datetime.now().isoformat()
        pages = []
        try:
            file_like = io.BytesIO(raw_data)
            wb = openpyxl.load_workbook(file_like, data_only=True, read_only=True)
            
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                sheet_rows = []
                for row in sheet.iter_rows(values_only=True):
                    # Convert row to cell value strings
                    row_values = [str(cell).strip() for cell in row if cell is not None]
                    if row_values:
                        sheet_rows.append(" | ".join(row_values))
                
                sheet_text = "\n".join(sheet_rows)
                if sheet_text.strip():
                    pages.append({
                        "text": sheet_text,
                        "metadata": {
                            "source": source_name,
                            "timestamp": timestamp,
                            "sheet_name": sheet_name
                        }
                    })
            logger.info(f"Successfully extracted {len(pages)} worksheets from {source_name}")
        except Exception as e:
            logger.error(f"Error parsing Excel workbook {source_name}: {e}")
        return pages
