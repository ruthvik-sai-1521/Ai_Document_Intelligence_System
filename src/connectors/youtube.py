import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, cast
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


def parse_vtt_text(vtt_text: str) -> List[Dict[str, Any]]:
    """Parse WebVTT content string into timestamped snippets."""
    snippets = []
    blocks = vtt_text.split('\n\n')
    time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})')
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        for i, line in enumerate(lines):
            match = time_pattern.search(line)
            if match:
                start_str = match.group(1)
                parts = start_str.split(':')
                if len(parts) == 3:
                    start_sec = float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
                else:
                    start_sec = float(parts[0])*60 + float(parts[1])
                
                text_lines = [l for l in lines[i+1:] if not l.isdigit() and '-->' not in l]
                text = ' '.join(text_lines)
                text = re.sub(r'<[^>]+>', '', text).strip()
                if text:
                    snippets.append({'text': text, 'start': start_sec, 'duration': 0.0})
                break
    return snippets


class YouTubeConnector(BaseConnector):
    def __init__(self, urls: List[str], languages: Optional[List[str]] = None):
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
                meta_title = (
                    soup.find("meta", property="og:title") or
                    soup.find("meta", attrs={"name": "title"})
                )
                if meta_title:
                    content_val = meta_title.get("content")
                    if isinstance(content_val, str) and content_val.strip():
                        title = content_val.strip()
                        if title.endswith(" - YouTube"):
                            title = title[:-10].strip()
                        return title
                
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
                    if title.endswith(" - YouTube"):
                        title = title[:-10].strip()
                    return title
        except Exception as e:
            logger.warning(f"Could not fetch video title for {video_id}: {e}")
        return f"YouTube Video ({video_id})"

    def _fetch_via_transcript_api(self, video_id: str) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Tier 1: Try youtube_transcript_api package.
        Returns (snippets, None) on success or (None, error_message) on failure.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            logger.info(f"Tier 1: Trying youtube_transcript_api for {video_id}...")

            # Support both instance and classmethod interfaces across versions
            raw_list = None
            if hasattr(YouTubeTranscriptApi, 'get_transcript'):
                try:
                    raw_list = YouTubeTranscriptApi.get_transcript(video_id, languages=self.languages)
                except Exception:
                    pass

            if not raw_list:
                api = YouTubeTranscriptApi()
                if hasattr(api, 'fetch'):
                    raw_list = api.fetch(video_id, languages=self.languages)

            if raw_list:
                snippets = []
                for item in raw_list:
                    if isinstance(item, dict):
                        text = str(item.get("text", "")).strip()
                        start = float(item.get("start", 0.0))
                        duration = float(item.get("duration", 0.0))
                    elif hasattr(item, "text") or hasattr(item, "start"):
                        text = str(getattr(item, "text", "")).strip()
                        start = float(getattr(item, "start", 0.0))
                        duration = float(getattr(item, "duration", 0.0))
                    elif hasattr(item, "get"):
                        text = str(item.get("text", "")).strip()
                        start = float(item.get("start", 0.0))
                        duration = float(item.get("duration", 0.0))
                    else:
                        continue
                    if text:
                        snippets.append({"text": text, "start": start, "duration": duration})
                if snippets:
                    logger.info(f"Tier 1 succeeded for {video_id} ({len(snippets)} snippets).")
                    return snippets, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.warning(f"Tier 1 youtube_transcript_api failed for {video_id}: {err}")
            return None, err
        return None, "Tier 1: no snippets returned by youtube_transcript_api"

    def _fetch_via_ytdlp(self, video_id: str) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Tier 2: Try yt-dlp to extract subtitles or auto-generated captions (handles IP blocks).
        Returns (snippets, None) on success or (None, error_message) on failure.
        """
        try:
            import yt_dlp
            logger.info(f"Tier 2: Trying yt-dlp caption extraction for {video_id}...")
            canonical_url = f"https://www.youtube.com/watch?v={video_id}"
            
            ydl_opts: Dict[str, Any] = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(canonical_url, download=False)
                subs = info.get('subtitles') or info.get('automatic_captions') or {}
                
                selected_track = None
                for lang in self.languages:
                    if lang in subs:
                        selected_track = subs[lang]
                        break
                if not selected_track and subs:
                    for k in subs:
                        if k.startswith('en'):
                            selected_track = subs[k]
                            break
                    if not selected_track:
                        selected_track = list(subs.values())[0]
                        
                if not selected_track:
                    msg = f"yt-dlp found no subtitle tracks for {video_id}."
                    logger.warning(msg)
                    return None, msg

                # Prefer json3 format for precise timestamps
                fmt = next((f for f in selected_track if f.get('ext') == 'json3'), selected_track[0])
                resp = requests.get(fmt['url'], timeout=10)
                if resp.status_code != 200:
                    msg = f"yt-dlp subtitle download failed (HTTP {resp.status_code}) for {video_id}."
                    return None, msg

                snippets = []
                if fmt.get('ext') == 'json3' or 'json3' in fmt.get('url', ''):
                    data = resp.json()
                    for event in data.get('events', []):
                        if 'segs' in event:
                            text = ''.join([s.get('utf8', '') for s in event['segs']]).strip()
                            if text and text != '\n':
                                start = event.get('tStartMs', 0) / 1000.0
                                duration = event.get('dDurationMs', 0) / 1000.0
                                snippets.append({"text": text, "start": start, "duration": duration})
                else:
                    snippets = parse_vtt_text(resp.text)

                if snippets:
                    logger.info(f"Tier 2 yt-dlp succeeded for {video_id} ({len(snippets)} snippets).")
                    return snippets, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.warning(f"Tier 2 yt-dlp extraction failed for {video_id}: {err}")
            return None, err
        return None, "Tier 2: no snippets returned by yt-dlp"

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

            # Tier 1: youtube_transcript_api
            snippets, tier1_error = self._fetch_via_transcript_api(video_id)

            # Tier 2: yt-dlp fallback if Tier 1 returned nothing
            tier2_error: Optional[str] = None
            if not snippets:
                snippets, tier2_error = self._fetch_via_ytdlp(video_id)

            if not snippets:
                # Surface the real error so it reaches the user via st.error() in the UI.
                # Never silently discard transcript failures.
                combined = ";".join(filter(None, [tier1_error, tier2_error]))
                raise RuntimeError(
                    f"Could not retrieve transcript for YouTube video '{video_title}' "
                    f"({canonical_url}).\n"
                    f"Reason: {combined or 'Unknown error in both transcript tiers.'}"
                )
                
            formatted_lines = []
            formatted_snippets = []
            for item in snippets:
                text = item["text"]
                start = item["start"]
                duration = item.get("duration", 0.0)
                time_str = format_seconds(start)
                
                formatted_snippets.append({
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
                    "snippets": formatted_snippets
                }
            })
            
            logger.info(f"Successfully retrieved transcript for '{video_title}' ({len(formatted_snippets)} snippets).")
            
        return documents

