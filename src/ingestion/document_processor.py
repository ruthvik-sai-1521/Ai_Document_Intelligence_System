from pathlib import Path
from typing import List, Dict, Any, Union
import re
from parsers.factory import ParserFactory
from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.logger import setup_logger

logger = setup_logger(__name__)

class DocumentProcessor:
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        # Chunk sizes are measured in number of words
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

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
        Includes overlap configuration window.
        """
        chunks = []
        current_chunk_items = []
        current_length = 0
        is_structured_row = False

        def save_chunk():
            nonlocal current_chunk_items, current_length
            if current_chunk_items:
                chunk_str = " ".join([item["text"] for item in current_chunk_items])
                
                # Re-aggregate metadata dynamically
                pages_set = {item["page_num"] for item in current_chunk_items if item["page_num"] is not None}
                slides_set = {item["slide_num"] for item in current_chunk_items if item["slide_num"] is not None}
                sheets_set = {item["sheet_name"] for item in current_chunk_items if item["sheet_name"] is not None}
                rows_set = {item["row_num"] for item in current_chunk_items if item["row_num"] is not None}
                
                ts_val = None
                for item in reversed(current_chunk_items):
                    if item["timestamp"] is not None:
                        ts_val = item["timestamp"]
                        break
                        
                is_ocr_val = any(item["is_ocr"] for item in current_chunk_items)
                
                ocr_meta_val = []
                for item in current_chunk_items:
                    if item["ocr_metadata"]:
                        ocr_meta_val.extend(item["ocr_metadata"])
                        
                meta = {
                    "source": source_id
                }
                if pages_set:
                    meta["page_number"] = self.format_page_numbers(pages_set)
                if slides_set:
                    meta["slide_number"] = list(slides_set)[0] if len(slides_set) == 1 else sorted(list(slides_set))
                if sheets_set:
                    meta["sheet_name"] = list(sheets_set)[0] if len(sheets_set) == 1 else sorted(list(sheets_set))
                if rows_set:
                    meta["row_number"] = list(rows_set)[0] if len(rows_set) == 1 else sorted(list(rows_set))
                if ts_val:
                    meta["timestamp"] = ts_val
                if is_ocr_val:
                    meta["is_ocr"] = True
                if ocr_meta_val:
                    meta["ocr_metadata"] = ocr_meta_val
                if user_id:
                    meta["user_id"] = user_id
                    
                # Carry over any custom keys from the composing pages
                for item in current_chunk_items:
                    p_meta = item.get("page_meta", {})
                    for k, v in p_meta.items():
                        if k not in meta and k not in ["page_number", "slide_number", "sheet_name", "row_number"]:
                            meta[k] = v
                            
                chunks.append({
                    "text": chunk_str,
                    "metadata": meta
                })
                
                # Overlap logic
                retained_items = []
                retained_words = 0
                if self.chunk_overlap > 0 and not is_structured_row:
                    for item in reversed(current_chunk_items):
                        if retained_words + item["word_count"] > self.max_chunk_size - self.min_chunk_size:
                            break
                        retained_items.append(item)
                        retained_words += item["word_count"]
                        if retained_words >= self.chunk_overlap:
                            break
                    retained_items.reverse()
                
                current_chunk_items = retained_items
                current_length = retained_words

        for page in pages:
            page_meta = page.get("metadata", {})
            page_num = page_meta.get("page_number")
            slide_num = page_meta.get("slide_number")
            sheet_name = page_meta.get("sheet_name")
            row_num = page_meta.get("row_number")
            timestamp = page_meta.get("timestamp")
            is_ocr = page_meta.get("is_ocr", False)
            ocr_metadata = page_meta.get("ocr_metadata", [])
            
            # Identify structured row source type
            is_structured_row = page_meta.get("source_type") == "structured_row"
            
            def add_item(text_block: str, word_count: int):
                current_chunk_items.append({
                    "text": text_block,
                    "word_count": word_count,
                    "page_num": page_num,
                    "slide_num": slide_num,
                    "sheet_name": sheet_name,
                    "row_num": row_num,
                    "timestamp": timestamp,
                    "is_ocr": is_ocr,
                    "ocr_metadata": ocr_metadata,
                    "page_meta": page_meta
                })

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
                                add_item(sentence, sentence_words)
                                current_length += sentence_words
                        else:
                            add_item(para, para_words)
                            current_length += para_words
                    else:
                        # Need more words, but adding para exceeds max. Split into sentences.
                        sentences = self.split_into_sentences(para)
                        for sentence in sentences:
                            sentence_words = len(sentence.split())
                            if current_length + sentence_words > self.max_chunk_size and current_length >= self.min_chunk_size:
                                save_chunk()
                            add_item(sentence, sentence_words)
                            current_length += sentence_words
                else:
                    add_item(para, para_words)
                    current_length += para_words
                    
            if is_structured_row:
                save_chunk()

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

    def process_raw_data(
        self,
        raw_data: bytes,
        source_name: str,
        extension: str,
        user_id: str = None,
        extra_metadata: dict = None
    ) -> List[Dict[str, Any]]:
        """
        Process in-memory raw data directly using factory parsers and run smart chunking.
        
        Args:
            raw_data:       Raw bytes of the file content.
            source_name:    Identifier string used as the chunk source (e.g. URL or filename).
            extension:      File extension (e.g. '.py', '.html').
            user_id:        Optional user identifier for chunk scoping.
            extra_metadata: Optional dict of additional metadata to merge into every chunk
                            (e.g. GitHub file_path, folder_hierarchy, repository fields).
        """
        try:
            parser = ParserFactory.get_parser(extension)
            pages = parser.parse(raw_data, {"source": source_name})
        except Exception as e:
            logger.error(f"Failed to process raw data {source_name}: {e}")
            raise e

        chunks = self.smart_chunking(pages, source_id=source_name, user_id=user_id)

        # Merge extra metadata into every chunk, preserving specific chunk keys
        if extra_metadata:
            for chunk in chunks:
                chunk_meta = chunk.setdefault("metadata", {})
                for k, v in extra_metadata.items():
                    if k not in chunk_meta:
                        chunk_meta[k] = v
                
            # If source is YouTube, derive start/end timestamps and clickable video URL
            if extra_metadata.get("source_type") == "youtube":
                video_id = extra_metadata.get("video_id")
                for chunk in chunks:
                    matches = re.findall(r'\[(\d{2}:\d{2}(?::\d{2})?)\]', chunk.get("text", ""))
                    if matches:
                        start_str = matches[0]
                        end_str = matches[-1]
                        
                        parts = [int(p) for p in start_str.split(":")]
                        start_sec = parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
                        
                        chunk["metadata"]["start_time"] = start_sec
                        chunk["metadata"]["start_formatted"] = start_str
                        chunk["metadata"]["end_formatted"] = end_str
                        chunk["metadata"]["formatted_time_range"] = f"[{start_str} - {end_str}]"
                        if video_id:
                            chunk["metadata"]["video_url_timestamped"] = f"https://www.youtube.com/watch?v={video_id}&t={start_sec}s"

        return chunks
