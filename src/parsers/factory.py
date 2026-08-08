from src.parsers.base import BaseParser
from src.parsers.pdf import PDFParser
from src.parsers.txt import TXTParser
from src.parsers.docx import DocxParser
from src.parsers.pptx import PptxParser
from src.parsers.xlsx import XlsxParser
from src.parsers.csv import CsvParser
from src.parsers.markdown import MarkdownParser
from src.parsers.html import HtmlParser
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class ParserFactory:
    # Pre-instantiate stateless parsers
    _parsers = {
        ".pdf": PDFParser(),
        ".txt": TXTParser(),
        ".docx": DocxParser(),
        ".pptx": PptxParser(),
        ".xlsx": XlsxParser(),
        ".csv": CsvParser(),
        ".md": MarkdownParser(),
        ".html": HtmlParser(),
        ".htm": HtmlParser(),
    }

    @classmethod
    def get_parser(cls, extension: str) -> BaseParser:
        """
        Returns the appropriate BaseParser instance based on file extension.
        
        Args:
            extension: The file extension starting with a dot (e.g. '.pdf', '.txt').
            
        Returns:
            An instance of BaseParser.
            
        Raises:
            ValueError: If the file extension is unsupported.
        """
        ext_lower = extension.lower().strip()
        parser = cls._parsers.get(ext_lower)
        if not parser:
            logger.error(f"Unsupported file extension: '{extension}'")
            raise ValueError(f"Unsupported file extension: '{extension}'")
        return parser
