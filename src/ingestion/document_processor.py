import PyPDF2
from pathlib import Path
from typing import List, Dict, Any, Union
import re
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class DocumentProcessor:
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 500):
        # Chunk sizes are measured in number of words
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def extract_pages_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from a PDF, preserving page numbers."""
        pages = []
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages):
                    extracted = page.extract_text()
                    if extracted:
                        pages.append({"page_num": i + 1, "text": extracted})
            logger.info(f"Successfully extracted {len(pages)} pages from {pdf_path}")
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {e}")
        return pages

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

    def smart_chunking(self, pages: List[Dict[str, Any]], source_id: str) -> List[Dict[str, Any]]:
        """
        Dynamically chunk text (400-800 words), preserving paragraphs and sentences.
        Adds source and page_number metadata.
        """
        chunks = []
        current_chunk_text = []
        current_length = 0
        current_pages = set()

        def save_chunk():
            nonlocal current_chunk_text, current_length, current_pages
            if current_chunk_text:
                chunks.append({
                    "text": " ".join(current_chunk_text),
                    "metadata": {
                        "source": source_id,
                        "page_number": self.format_page_numbers(current_pages)
                    }
                })
                current_chunk_text = []
                current_length = 0
                current_pages = set()

        for page in pages:
            page_num = page["page_num"]
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
                                current_pages.add(page_num)
                        else:
                            current_chunk_text.append(para)
                            current_length += para_words
                            current_pages.add(page_num)
                    else:
                        # Need more words, but adding para exceeds max. Split into sentences.
                        sentences = self.split_into_sentences(para)
                        for sentence in sentences:
                            sentence_words = len(sentence.split())
                            if current_length + sentence_words > self.max_chunk_size and current_length >= self.min_chunk_size:
                                save_chunk()
                            current_chunk_text.append(sentence)
                            current_length += sentence_words
                            current_pages.add(page_num)
                else:
                    current_chunk_text.append(para)
                    current_length += para_words
                    current_pages.add(page_num)

        # Add the last chunk if any
        save_chunk()
            
        logger.info(f"Chunked {source_id} into {len(chunks)} chunks.")
        return chunks

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single document from extraction to chunking."""
        path = Path(file_path)
        pages = []
        if path.suffix.lower() == '.pdf':
            pages = self.extract_pages_from_pdf(str(path))
        elif path.suffix.lower() == '.txt':
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
                # For TXT, we don't have pages, just assign page 1
                pages = [{"page_num": 1, "text": text}]
        else:
            logger.error(f"Unsupported file type: {path.suffix}")
            raise ValueError(f"Unsupported file type: {path.suffix}")
            
        chunks = self.smart_chunking(pages, source_id=path.name)
        return chunks

    def process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents and combine chunks."""
        all_chunks = []
        for path in file_paths:
            logger.info(f"Processing document: {path}")
            chunks = self.process_document(path)
            all_chunks.extend(chunks)
        return all_chunks
