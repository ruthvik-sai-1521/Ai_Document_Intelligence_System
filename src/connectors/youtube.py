import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from connectors.base import BaseConnector
from core.logger import setup_logger

logger = setup_logger(__name__)

# Regex pattern for YouTube video ID extraction
YOUTUBE_ID_PATTERN = re.compile(r'(?:v=|\/|be\/|embed\/|shorts\/|^)([a-zA-Z0-9_-]{11})(?:[&?]|$)')


def extract_video_id(url: str) -> Optional[str]:
    """Extract 11-character YouTube video ID from various URL formats or raw ID."""
    if not url:
        return None
    url_clean = url.strip()
    match = YOUTUBE_ID_PATTERN.search(url_clean)
    if match:
        return match.group(1)
    return None


def format_seconds(seconds: float) -> str:
    """Format total seconds into MM:SS or HH:MM:SS."""
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class YouTubeConnector(BaseConnector):
    def __init__(self, urls: List[str], languages: List[str] = None):
        """
        Args:
            urls:      List of YouTube video URLs or shorthand video IDs.
            languages: Priority language codes for transcript retrieval.
        """
        self.urls = urls
        self.languages = languages or ["en", "en-US", "en-GB"]

    def fetch_video_title(self, video_id: str) -> str:
        """Fetch video title using YouTube oEmbed API or fallback watch page HTML meta tags."""
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        try:
            logger.info(f"Attempting to fetch title via YouTube oEmbed API for {video_id}...")
            response = requests.get(oembed_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "title" in data and data["title"].strip():
                    title = data["title"].strip()
                    logger.info(f"Successfully fetched title '{title}' via oEmbed API.")
                    return title
        except Exception as e:
            logger.debug(f"oEmbed title fetch failed for {video_id}: {e}")

        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            logger.info(f"Falling back to page scraping title for {video_id}...")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(watch_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Check meta tag name="title" or property="og:title"
                meta_title = (
                    soup.find("meta", property="og:title") or
                    soup.find("meta", attrs={"name": "title"})
                )
                if meta_title and meta_title.get("content"):
                    title = meta_title["content"].strip()
                    if title.endswith(" - YouTube"):
                        title = title[:-10].strip()
                    return title
                
                # Fallback to page title tag
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
                    if title.endswith(" - YouTube"):
                        title = title[:-10].strip()
                    return title
        except Exception as e:
            logger.warning(f"Could not fetch video title for {video_id}: {e}")
        return f"YouTube Video ({video_id})"

    def fetch_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieves YouTube transcripts and formats documents with timestamp metadata.
        Returns:
            List of document dictionaries.
        """
        documents = []
        
        for raw_url in self.urls:
            raw_url = raw_url.strip()
            if not raw_url:
                continue
                
            video_id = extract_video_id(raw_url)
            if not video_id:
                logger.warning(f"Could not parse valid YouTube video ID from URL: {raw_url}")
                continue
                
            canonical_url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Processing YouTube Video ID: {video_id} ({canonical_url})")
            
            video_title = self.fetch_video_title(video_id)
            
            try:
                # Fetch transcript
                api = YouTubeTranscriptApi()
                transcript_api_list = api.fetch(video_id, languages=self.languages)
                
                if not transcript_api_list:
                    logger.warning(f"No transcript found for video ID: {video_id}")
                    continue
                    
                # Format transcript snippets with timestamps
                snippets = []
                formatted_lines = []
                
                for item in transcript_api_list:
                    if hasattr(item, "text"):
                        text = getattr(item, "text", "").strip()
                        start = getattr(item, "start", 0.0)
                        duration = getattr(item, "duration", 0.0)
                    else:
                        text = item.get("text", "").strip()
                        start = item.get("start", 0.0)
                        duration = item.get("duration", 0.0)
                    time_str = format_seconds(start)
                    
                    if text:
                        snippets.append({
                            "text": text,
                            "start": start,
                            "duration": duration,
                            "formatted_time": time_str
                        })
                        formatted_lines.append(f"[{time_str}] {text}")
                        
                full_transcript_text = "\n".join(formatted_lines)
                
                documents.append({
                    "raw_data": full_transcript_text.encode("utf-8"),
                    "source": canonical_url,
                    "extension": ".txt",
                    "metadata": {
                        "source_type": "youtube",
                        "video_id": video_id,
                        "video_title": video_title,
                        "video_url": canonical_url,
                        "snippets": snippets
                    }
                })
                
                logger.info(f"Successfully retrieved transcript for '{video_title}' ({len(snippets)} snippets).")
                
            except Exception as e:
                logger.error(f"Failed to retrieve transcript for YouTube video {video_id}: {e}")
                
        return documents
