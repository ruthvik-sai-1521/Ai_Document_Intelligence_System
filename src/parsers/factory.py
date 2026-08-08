from src.parsers.base import BaseParser
from src.parsers.pdf import PDFParser
from src.parsers.txt import TXTParser
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class ParserFactory:
    # Pre-instantiate stateless parsers
    _parsers = {
        ".pdf": PDFParser(),
        ".txt": TXTParser(),
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
