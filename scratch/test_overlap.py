import sys
from pathlib import Path

# Add project root and src directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.ingestion.document_processor import DocumentProcessor

# max_chunk_size is 10 words, chunk_overlap is 4 words
processor = DocumentProcessor(min_chunk_size=3, max_chunk_size=10, chunk_overlap=4)

# We have multiple small paragraphs so they are added as separate items
pages = [
    {
        "text": "Sentence one.\n\nSentence two.\n\nSentence three.\n\nSentence four.",
        "metadata": {"page_number": 1, "timestamp": "2026-08-10T12:00:00"}
    },
    {
        "text": "Sentence five.\n\nSentence six.\n\nSentence seven.",
        "metadata": {"page_number": 2, "timestamp": "2026-08-10T12:01:00"}
    }
]

chunks = processor.smart_chunking(pages, "test_doc")
for i, c in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(f"Text: {c['text']}")
    print(f"Metadata: {c['metadata']}")
