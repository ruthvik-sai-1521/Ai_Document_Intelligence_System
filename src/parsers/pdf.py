import io
import PyPDF2
import fitz  # PyMuPDF
from datetime import datetime
from typing import List, Dict, Any
from parsers.base import BaseParser
from core.logger import setup_logger

logger = setup_logger(__name__)

class PDFParser(BaseParser):
    def __init__(self, ocr_engine=None):
        self._ocr_engine = ocr_engine
        
    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            # Lazily load OCR Engine to save memory/computation on digital-only PDF parses
            from ocr.easy_ocr import EasyOCREngine
            from core.config import OCR_LANGUAGES
            self._ocr_engine = EasyOCREngine(languages=OCR_LANGUAGES)
        return self._ocr_engine

    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse raw PDF byte stream, extracting digital text or falling back to OCR if scanned."""
        pages = []
        source_name = metadata.get("source", "Unknown PDF")
        timestamp = datetime.now().isoformat()
        
        try:
            file_like = io.BytesIO(raw_data)
            reader = PyPDF2.PdfReader(file_like)
            
            fitz_doc = None
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                
                # Scanned check: less than 20 characters
                is_scanned = not extracted or len(extracted.strip()) < 20
                
                if not is_scanned:
                    pages.append({
                        "text": extracted,
                        "metadata": {
                            "source": source_name,
                            "timestamp": timestamp,
                            "page_number": i + 1
                        }
                    })
                else:
                    logger.info(f"Page {i + 1} in {source_name} appears to be scanned. Falling back to OCR...")
                    if fitz_doc is None:
                        fitz_doc = fitz.open(stream=raw_data, filetype="pdf")
                    
                    fitz_page = fitz_doc[i]
                    # Render page at 150 DPI for optimal OCR readability
                    pix = fitz_page.get_pixmap(dpi=150)
                    png_bytes = pix.tobytes("png")
                    
                    ocr_blocks = self.ocr_engine.extract_text_with_metadata(png_bytes)
                    ocr_text = " ".join([block["text"] for block in ocr_blocks])
                    
                    pages.append({
                        "text": ocr_text,
                        "metadata": {
                            "source": source_name,
                            "timestamp": timestamp,
                            "page_number": i + 1,
                            "is_ocr": True,
                            "ocr_metadata": ocr_blocks
                        }
                    })
            
            if fitz_doc:
                fitz_doc.close()
                
            logger.info(f"Successfully processed {len(pages)} pages from PDF {source_name}")
        except Exception as e:
            logger.error(f"Error reading PDF {source_name}: {e}")
        return pages
