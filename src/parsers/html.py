from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Any
from src.parsers.base import BaseParser
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class HtmlParser(BaseParser):
    def parse(self, raw_data: bytes, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse HTML bytes and extract clean body text."""
        source_name = metadata.get("source", "Unknown HTML File")
        timestamp = datetime.now().isoformat()
        try:
            soup = BeautifulSoup(raw_data, "html.parser")
            
            # Remove scripts, styles, and standard layout containers
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
                
            # Decompose tags matching ad, sidebar, navigation, and popup classes or IDs
            import re
            pattern = re.compile(
                r'ad-|advertisement|sidebar|social-share|share-bar|share-buttons|'
                r'social-links|footer-menu|nav-menu|cookie-consent|banner|popup', 
                re.I
            )
            for element in soup.find_all(attrs={"class": pattern}):
                element.decompose()
            for element in soup.find_all(attrs={"id": pattern}):
                element.decompose()
                
            # Extract clean paragraphs text(separator="\n")
            text = soup.get_text(separator="\n")
            
            # Clean up whitespace and newlines
            lines = [line.strip() for line in text.splitlines()]
            clean_text = "\n".join([line for line in lines if line])
            
            logger.info(f"Successfully extracted clean text from HTML {source_name}")
            return [{
                "text": clean_text,
                "metadata": {
                    "source": source_name,
                    "timestamp": timestamp,
                    "page_number": 1
                }
            }]
        except Exception as e:
            logger.error(f"Error parsing HTML file {source_name}: {e}")
            return []
