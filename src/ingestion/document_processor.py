from pathlib import Path
from typing import List, Dict, Any, Union
import re
from src.parsers.factory import ParserFactory
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class DocumentProcessor:
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 500):
        # Chunk sizes are measured in number of words
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size


    def clean_text(self, text: str) -> str:
        """Basic text cleaning, preserving paragraph boundaries."""
        # Replace 3 or more newlines with 2 newlines to preserve paragraphs
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences without breaking them."""
        # A simple regex to split by sentence boundaries (.!?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def format_page_numbers(self, pages: set) -> Union[int, List[int]]:
        """Format the page numbers for metadata."""
        if len(pages) == 1:
            return list(pages)[0]
        return sorted(list(pages))

    def smart_chunking(self, pages: List[Dict[str, Any]], source_id: str, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Dynamically chunk text (400-800 words), preserving paragraphs and sentences.
        Adds source, page/slide/sheet/row, timestamp, OCR metadata, and user_id metadata.
        """
        chunks = []
        current_chunk_text = []
        current_length = 0
        current_pages = set()
        current_slides = set()
        current_sheets = set()
        current_rows = set()
        current_timestamp = None
        current_is_ocr = False
        current_ocr_metadata = []

        def save_chunk():
            nonlocal current_chunk_text, current_length, current_pages, current_slides, current_sheets, current_rows, current_timestamp, current_is_ocr, current_ocr_metadata
            if current_chunk_text:
                meta = {
                    "source": source_id
                }
                if current_pages:
                    meta["page_number"] = self.format_page_numbers(current_pages)
                if current_slides:
                    meta["slide_number"] = list(current_slides)[0] if len(current_slides) == 1 else sorted(list(current_slides))
                if current_sheets:
                    meta["sheet_name"] = list(current_sheets)[0] if len(current_sheets) == 1 else sorted(list(current_sheets))
                if current_rows:
                    meta["row_number"] = list(current_rows)[0] if len(current_rows) == 1 else sorted(list(current_rows))
                if current_timestamp:
                    meta["timestamp"] = current_timestamp
                if current_is_ocr:
                    meta["is_ocr"] = True
                if current_ocr_metadata:
                    meta["ocr_metadata"] = current_ocr_metadata
                if user_id:
                    meta["user_id"] = user_id
                    
                chunks.append({
                    "text": " ".join(current_chunk_text),
                    "metadata": meta
                })
                current_chunk_text = []
                current_length = 0
                current_pages = set()
                current_slides = set()
                current_sheets = set()
                current_rows = set()
                current_timestamp = None
                current_is_ocr = False
                current_ocr_metadata = []

        for page in pages:
            page_meta = page.get("metadata", {})
            page_num = page_meta.get("page_number")
            slide_num = page_meta.get("slide_number")
            sheet_name = page_meta.get("sheet_name")
            row_num = page_meta.get("row_number")
            timestamp = page_meta.get("timestamp")
            is_ocr = page_meta.get("is_ocr", False)
            ocr_metadata = page_meta.get("ocr_metadata", [])
            
            def record_metadata():
                nonlocal current_timestamp, current_is_ocr, current_ocr_metadata
                if page_num is not None:
                    current_pages.add(page_num)
                if slide_num is not None:
                    current_slides.add(slide_num)
                if sheet_name is not None:
                    current_sheets.add(sheet_name)
                if row_num is not None:
                    current_rows.add(row_num)
                if timestamp is not None:
                    current_timestamp = timestamp
                if is_ocr:
                    current_is_ocr = True
                if ocr_metadata:
                    current_ocr_metadata.extend(ocr_metadata)

            cleaned_text = self.clean_text(page["text"])
            paragraphs = cleaned_text.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Replace inner newlines with space to form a continuous paragraph
                para = re.sub(r'\n', ' ', para)
                para_words = len(para.split())
                
                if current_length + para_words > self.max_chunk_size:
                    if current_length >= self.min_chunk_size:
                        save_chunk()
                        
                        # Process the new paragraph on its own
                        if para_words > self.max_chunk_size:
                            # Split into sentences
                            sentences = self.split_into_sentences(para)
                            for sentence in sentences:
                                sentence_words = len(sentence.split())
                                if current_length + sentence_words > self.max_chunk_size and current_length >= self.min_chunk_size:
                                    save_chunk()
                                current_chunk_text.append(sentence)
                                current_length += sentence_words
                                record_metadata()
                        else:
                            current_chunk_text.append(para)
                            current_length += para_words
                            record_metadata()
                    else:
                        # Need more words, but adding para exceeds max. Split into sentences.
                        sentences = self.split_into_sentences(para)
                        for sentence in sentences:
                            sentence_words = len(sentence.split())
                            if current_length + sentence_words > self.max_chunk_size and current_length >= self.min_chunk_size:
                                save_chunk()
                            current_chunk_text.append(sentence)
                            current_length += sentence_words
                            record_metadata()
                else:
                    current_chunk_text.append(para)
                    current_length += para_words
                    record_metadata()

        # Add the last chunk if any
        save_chunk()
            
        logger.info(f"Chunked {source_id} into {len(chunks)} chunks.")
        return chunks

    def process_document(self, file_path: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Process a single document from extraction to chunking."""
        path = Path(file_path)
        try:
            with open(path, "rb") as file:
                raw_data = file.read()
            parser = ParserFactory.get_parser(path.suffix)
            pages = parser.parse(raw_data, {"source": path.name})
        except Exception as e:
            logger.error(f"Failed to process document {path.name}: {e}")
            raise e
            
        chunks = self.smart_chunking(pages, source_id=path.name, user_id=user_id)
        return chunks

    def process_documents(self, file_paths: List[str], user_id: str = None) -> List[Dict[str, Any]]:
        """Process multiple documents and combine chunks."""
        all_chunks = []
        for path in file_paths:
            logger.info(f"Processing document: {path}")
            chunks = self.process_document(path, user_id=user_id)
            all_chunks.extend(chunks)
        return all_chunks

    def process_raw_data(self, raw_data: bytes, source_name: str, extension: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Process in-memory raw data directly using factory parsers and run smart chunking."""
        try:
            parser = ParserFactory.get_parser(extension)
            pages = parser.parse(raw_data, {"source": source_name})
        except Exception as e:
            logger.error(f"Failed to process raw data {source_name}: {e}")
            raise e
            
        chunks = self.smart_chunking(pages, source_id=source_name, user_id=user_id)
        return chunks

