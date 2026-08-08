import urllib.robotparser
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Set
import requests
from bs4 import BeautifulSoup
from src.connectors.base import BaseConnector
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class WebConnector(BaseConnector):
    def __init__(
        self,
        urls: List[str],
        max_depth: int = 1,
        use_sitemap: bool = False,
        respect_robots: bool = True,
        timeout: int = 10
    ):
        self.start_urls = urls
        self.max_depth = max_depth
        self.use_sitemap = use_sitemap
        self.respect_robots = respect_robots
        self.timeout = timeout
        
    def fetch_documents(self) -> List[Dict[str, Any]]:
        """
        Crawls start URLs recursively up to max_depth.
        Returns:
            List of dictionaries containing raw html bytes and URL metadata.
        """
        documents = []
        visited_urls = set()
        urls_to_visit = []
        
        # Enqueue initial URLs with depth 1
        for url in self.start_urls:
            url = url.strip()
            if url:
                urls_to_visit.append((url, 1))
            
        # Sitemap URL discovery
        if self.use_sitemap:
            for url in self.start_urls:
                sitemap_urls = self._discover_sitemap_urls(url)
                for s_url in sitemap_urls:
                    if s_url not in visited_urls:
                        urls_to_visit.append((s_url, 1))

        # BFS Crawling Queue
        while urls_to_visit:
            current_url, depth = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
            visited_urls.add(current_url)
            
            # Respect robots.txt
            if self.respect_robots and not self._is_allowed_by_robots(current_url):
                logger.warning(f"Robots.txt restricted access to URL: {current_url}")
                continue
                
            try:
                logger.info(f"Crawling URL (Depth {depth}/{self.max_depth}): {current_url}")
                response = requests.get(
                    current_url, 
                    timeout=self.timeout, 
                    headers={"User-Agent": "DocuMind-Crawler/1.0"}
                )
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch {current_url}: Status code {response.status_code}")
                    continue
                    
                documents.append({
                    "raw_data": response.content,
                    "source": current_url,
                    "extension": ".html"
                })
                
                # If we are within max depth, extract same-domain internal links
                if depth < self.max_depth:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    for anchor in soup.find_all('a', href=True):
                        href = anchor['href']
                        # Build absolute URL from relative path
                        full_url = urljoin(current_url, href)
                        # Strip hashes/fragment identifiers
                        full_url = full_url.split('#')[0].strip()
                        
                        if (self._is_same_domain(current_url, full_url) and 
                                full_url not in visited_urls and 
                                full_url.startswith("http")):
                            urls_to_visit.append((full_url, depth + 1))
                            
            except Exception as e:
                logger.error(f"Error crawling {current_url}: {e}")
                
        logger.info(f"Crawl finished. Ingested {len(documents)} pages.")
        return documents

    def _is_same_domain(self, start_url: str, target_url: str) -> bool:
        """Checks if target URL shares the same domain as starting URL."""
        start_netloc = urlparse(start_url).netloc
        target_netloc = urlparse(target_url).netloc
        return start_netloc == target_netloc

    def _is_allowed_by_robots(self, url: str) -> bool:
        """Fetch and check robots.txt crawl rules."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            rp = urllib.robotparser.RobotFileParser()
            response = requests.get(
                robots_url, 
                timeout=5, 
                headers={"User-Agent": "DocuMind-Crawler/1.0"}
            )
            if response.status_code == 200:
                rp.parse(response.text.splitlines())
                return rp.can_fetch("DocuMind-Crawler/1.0", url) or rp.can_fetch("*", url)
            elif response.status_code == 404:
                return True  # If robots.txt doesn't exist, all paths allowed
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
        return True  # Allow fallback if checking fails

    def _discover_sitemap_urls(self, base_url: str) -> List[str]:
        """Fetch sitemap.xml and parse target URLs."""
        urls = []
        try:
            parsed_url = urlparse(base_url)
            sitemap_url = f"{parsed_url.scheme}://{parsed_url.netloc}/sitemap.xml"
            logger.info(f"Checking for sitemap at: {sitemap_url}")
            
            response = requests.get(
                sitemap_url, 
                timeout=5, 
                headers={"User-Agent": "DocuMind-Crawler/1.0"}
            )
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                # Handle XML namespace mappings
                namespace = ""
                if root.tag.startswith("{"):
                    namespace = root.tag.split("}")[0] + "}"
                
                for loc in root.findall(f".//{namespace}loc"):
                    if loc.text:
                        urls.append(loc.text.strip())
                logger.info(f"Discovered {len(urls)} URLs inside sitemap.xml")
        except Exception as e:
            logger.warning(f"Error parsing sitemap for {base_url}: {e}")
        return urls
