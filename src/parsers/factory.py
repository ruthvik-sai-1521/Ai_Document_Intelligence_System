try:
    from src.parsers.base import BaseParser
    from src.parsers.pdf import PDFParser
    from src.parsers.txt import TXTParser
    from src.parsers.docx import DocxParser
    from src.parsers.pptx import PptxParser
    from src.parsers.xlsx import XlsxParser
    from src.parsers.csv import CsvParser
    from src.parsers.markdown import MarkdownParser
    from src.parsers.html import HtmlParser
    from src.parsers.image import ImageParser
    from src.parsers.structured import StructuredDataParser
    from src.core.logger import setup_logger
except ImportError:
    from parsers.base import BaseParser
    from parsers.pdf import PDFParser
    from parsers.txt import TXTParser
    from parsers.docx import DocxParser
    from parsers.pptx import PptxParser
    from parsers.xlsx import XlsxParser
    from parsers.csv import CsvParser
    from parsers.markdown import MarkdownParser
    from parsers.html import HtmlParser
    from parsers.image import ImageParser
    from parsers.structured import StructuredDataParser
    from core.logger import setup_logger

logger = setup_logger(__name__)

# Singleton TXTParser reused for code and config extensions
_txt = TXTParser()

class ParserFactory:
    # Pre-instantiate stateless parsers
    _parsers = {
        # Document formats
        ".pdf":  PDFParser(),
        ".txt":  _txt,
        ".docx": DocxParser(),
        ".pptx": PptxParser(),
        ".xlsx": XlsxParser(),
        ".csv":  CsvParser(),
        ".md":   MarkdownParser(),
        ".html": HtmlParser(),
        ".htm":  HtmlParser(),
        ".json": StructuredDataParser(),
        # Image formats
        ".png":  ImageParser(),
        ".jpg":  ImageParser(),
        ".jpeg": ImageParser(),
        ".tiff": ImageParser(),
        ".tif":  ImageParser(),
        # Source code (plain text extraction)
        ".py":    _txt,
        ".java":  _txt,
        ".js":    _txt,
        ".ts":    _txt,
        ".jsx":   _txt,
        ".tsx":   _txt,
        ".c":     _txt,
        ".cpp":   _txt,
        ".h":     _txt,
        ".cs":    _txt,
        ".go":    _txt,
        ".rb":    _txt,
        ".rs":    _txt,
        ".swift": _txt,
        ".kt":    _txt,
        ".php":   _txt,
        ".scala": _txt,
        ".sh":    _txt,
        ".bash":  _txt,
        ".css":   _txt,
        ".rst":   _txt,
        # Configuration formats
        ".yaml":  _txt,
        ".yml":   _txt,
        ".xml":   _txt,
        ".ini":   _txt,
        ".conf":  _txt,
        ".toml":  _txt,
        ".cfg":   _txt,
        ".env":   _txt,
    }

    @classmethod
    def get_parser(cls, extension: str) -> BaseParser:
        """
        Returns the appropriate BaseParser for the given file extension.
        Falls back to TXTParser for unrecognized or missing extensions.

        Args:
            extension: File extension starting with a dot (e.g. '.pdf', '.py').

        Returns:
            An instance of BaseParser.
        """
        ext_lower = extension.lower().strip() if extension else ""
        parser = cls._parsers.get(ext_lower)
        if not parser:
            logger.debug(f"Unknown extension '{ext_lower}', falling back to TXTParser.")
            return _txt
        return parser
