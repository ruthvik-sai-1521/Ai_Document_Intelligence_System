"""
YouTube Ingestion Pipeline — Full Diagnostic Trace
Tests every stage without modifying any implementation code.
Run with: .venv\Scripts\python.exe scratch/youtube_pipeline_trace.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

DIVIDER = "=" * 80
MINI = "-" * 60

# ── Test URL ─────────────────────────────────────────────────────────
# Rick Astley - "Never Gonna Give You Up" — guaranteed public video with captions
TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

print(DIVIDER)
print("YOUTUBE INGESTION PIPELINE DIAGNOSTIC TRACE")
print(DIVIDER)
print(f"Test URL: {TEST_URL}\n")

# ─────────────────────────────────────────────────────────────────────
# CHECK 1: URL Parsing and Video ID Extraction
# ─────────────────────────────────────────────────────────────────────
print(MINI)
print("CHECK 1: URL VALIDATION AND VIDEO ID EXTRACTION")
print(MINI)
from connectors.youtube import extract_video_id, YOUTUBE_ID_PATTERN

video_id = extract_video_id(TEST_URL)
print(f"  Input URL  : {TEST_URL}")
print(f"  Regex used : {YOUTUBE_ID_PATTERN.pattern}")
print(f"  Video ID   : {video_id!r}")
if video_id:
    print(f"  RESULT     : PASS - video_id={video_id!r} (11 chars: {len(video_id)})")
else:
    print(f"  RESULT     : FAIL - could not extract video ID")

# ─────────────────────────────────────────────────────────────────────
# CHECK 2: Library Installation
# ─────────────────────────────────────────────────────────────────────
print(f"\n{MINI}")
print("CHECK 2: LIBRARY INSTALLATION STATUS")
print(MINI)

# youtube_transcript_api
try:
    import youtube_transcript_api as yta_mod
    print(f"  youtube-transcript-api : INSTALLED")
    import pkg_resources
    try:
        ver = pkg_resources.get_distribution("youtube-transcript-api").version
        print(f"  Version                : {ver}")
    except Exception:
        print(f"  Version                : unknown (no __version__ attr)")
except ImportError as e:
    print(f"  youtube-transcript-api : NOT INSTALLED -> {e}")

# yt-dlp
try:
    import yt_dlp
    print(f"  yt-dlp                 : INSTALLED version {yt_dlp.version.__version__}")
except ImportError as e:
    print(f"  yt-dlp                 : NOT INSTALLED -> {e}")

# ─────────────────────────────────────────────────────────────────────
# CHECK 3: YouTubeTranscriptApi class interface (version mismatch?)
# ─────────────────────────────────────────────────────────────────────
print(f"\n{MINI}")
print("CHECK 3: YouTubeTranscriptApi CLASS INTERFACE INSPECTION")
print(MINI)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    import inspect

    has_get_transcript = hasattr(YouTubeTranscriptApi, "get_transcript")
    has_fetch          = hasattr(YouTubeTranscriptApi, "fetch")
    has_list           = hasattr(YouTubeTranscriptApi, "list")

    fetch_is_classmethod  = isinstance(inspect.getattr_static(YouTubeTranscriptApi, "fetch", None), classmethod) if has_fetch else False
    fetch_is_staticmethod = isinstance(inspect.getattr_static(YouTubeTranscriptApi, "fetch", None), staticmethod) if has_fetch else False

    print(f"  has 'get_transcript'  : {has_get_transcript}  <- connector code checks this FIRST")
    print(f"  has 'fetch'           : {has_fetch}")
    print(f"  has 'list'            : {has_list}")
    print(f"  fetch is classmethod  : {fetch_is_classmethod}")
    print(f"  fetch is staticmethod : {fetch_is_staticmethod}")
    print(f"  fetch is INSTANCE method : {has_fetch and not fetch_is_classmethod and not fetch_is_staticmethod}")

    if has_fetch:
        sig = inspect.signature(YouTubeTranscriptApi.fetch)
        print(f"  fetch() signature     : {sig}")

    print()
    print("  CONNECTOR CODE LOGIC (youtube.py lines 124-133):")
    print("  -----------------------------------------------")
    print("  if hasattr(YouTubeTranscriptApi, 'get_transcript'):")
    print("      raw_list = YouTubeTranscriptApi.get_transcript(...)  # classmethod style")
    print()
    print("  if not raw_list:")
    print("      api = YouTubeTranscriptApi()")
    print("      if hasattr(api, 'fetch'):")
    print("          raw_list = api.fetch(video_id, languages=...)    # instance method style")
    print()

    if not has_get_transcript:
        print("  DIAGNOSIS: 'get_transcript' does NOT exist on v1.2.4")
        print("             -> First branch SKIPPED, falls to instance method path")
    if has_fetch and not fetch_is_classmethod:
        print("  DIAGNOSIS: 'fetch' is an INSTANCE method, not a classmethod")
        print("             -> api = YouTubeTranscriptApi() then api.fetch() is CORRECT path")
        print("             -> BUT: does api.fetch() accept 'languages=' kwarg?")
        sig = inspect.signature(YouTubeTranscriptApi.fetch)
        params = list(sig.parameters.keys())
        print(f"             -> fetch params: {params}")
        if "languages" in params:
            print("             -> 'languages' param EXISTS - OK")
        else:
            print("             -> 'languages' param MISSING - KWARG MISMATCH!")

except Exception as e:
    print(f"  FAILED to inspect: {type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────────────────
# CHECK 4: Simulate connector's Tier 1 _fetch_via_transcript_api
# ─────────────────────────────────────────────────────────────────────
print(f"\n{MINI}")
print("CHECK 4: SIMULATE TIER 1 TRANSCRIPT API CALL (connector logic exactly)")
print(MINI)

if not video_id:
    print("  SKIP - no video_id extracted")
else:
    print(f"  Calling with video_id={video_id!r}, languages=['en','en-US','en-GB']")
    languages = ["en", "en-US", "en-GB"]

    raw_list = None
    step_error = None

    # Exactly mirrors connector code lines 124-133
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        # Step A: try class method 'get_transcript'
        if hasattr(YouTubeTranscriptApi, 'get_transcript'):
            print("  Step A: get_transcript exists, calling as classmethod...")
            try:
                raw_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                print(f"  Step A: SUCCEEDED, got {len(raw_list)} items")
            except Exception as e:
                print(f"  Step A: FAILED: {type(e).__name__}: {e}")
        else:
            print("  Step A: SKIPPED - 'get_transcript' does not exist on this version")

        # Step B: instance method 'fetch'
        if not raw_list:
            print("  Step B: Trying instance method api.fetch()...")
            api = YouTubeTranscriptApi()
            if hasattr(api, 'fetch'):
                try:
                    raw_list = api.fetch(video_id, languages=languages)
                    print(f"  Step B: SUCCEEDED, type={type(raw_list).__name__}, len={len(list(raw_list)) if hasattr(raw_list, '__len__') else 'unknown'}")
                except TypeError as e:
                    print(f"  Step B: FAILED with TypeError: {e}")
                    step_error = e
                    # Try without languages kwarg
                    print("  Step B-retry: Trying api.fetch(video_id) without languages kwarg...")
                    try:
                        raw_list = api.fetch(video_id)
                        print(f"  Step B-retry: SUCCEEDED, type={type(raw_list).__name__}")
                    except Exception as e2:
                        print(f"  Step B-retry: ALSO FAILED: {type(e2).__name__}: {e2}")
                except Exception as e:
                    print(f"  Step B: FAILED: {type(e).__name__}: {str(e)[:300]}")
                    step_error = e
            else:
                print("  Step B: SKIPPED - 'fetch' not on instance")

    except ImportError as e:
        print(f"  IMPORT FAILED: {e}")

    print()
    if raw_list is not None:
        items = list(raw_list)
        print(f"  Tier 1 raw_list obtained: {len(items)} items")
        if items:
            item0 = items[0]
            print(f"  First item type : {type(item0).__name__}")
            print(f"  First item repr : {repr(item0)[:200]}")
            # Test connector's parsing code
            if isinstance(item0, dict):
                print(f"  Parsed as dict  : text={item0.get('text','')!r}, start={item0.get('start',0)}")
            elif hasattr(item0, 'text'):
                print(f"  Parsed as obj   : text={getattr(item0,'text','')!r}, start={getattr(item0,'start',0)}")
    else:
        print(f"  Tier 1 FAILED - raw_list is None")
        if step_error:
            print(f"  Last error: {type(step_error).__name__}: {step_error}")

# ─────────────────────────────────────────────────────────────────────
# CHECK 5: FetchedTranscript object structure (v1.2.4 returns object, not list)
# ─────────────────────────────────────────────────────────────────────
print(f"\n{MINI}")
print("CHECK 5: FetchedTranscript OBJECT STRUCTURE (v1.2.4 API)")
print(MINI)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    print(f"  Calling api.fetch({video_id!r})...")
    result = api.fetch(video_id)
    print(f"  Return type: {type(result).__name__}")
    print(f"  Module: {type(result).__module__}")
    print(f"  Has __iter__: {hasattr(result, '__iter__')}")
    print(f"  Has __len__: {hasattr(result, '__len__')}")
    print(f"  Has text attr (is it a list-like of snippets?): {hasattr(result, 'fetch')}")

    # Try to iterate
    snippets = list(result)
    print(f"  Iterable: YES, {len(snippets)} snippets")
    if snippets:
        s0 = snippets[0]
        print(f"  Snippet[0] type: {type(s0).__name__}")
        print(f"  Snippet[0] repr: {repr(s0)[:200]}")
        print(f"  isinstance dict: {isinstance(s0, dict)}")
        print(f"  hasattr .text  : {hasattr(s0, 'text')}")
        print(f"  hasattr .start : {hasattr(s0, 'start')}")
        if hasattr(s0, 'text'):
            print(f"  s0.text = {s0.text!r}")
        if hasattr(s0, 'start'):
            print(f"  s0.start = {s0.start!r}")
        print(f"\n  Connector handles this type? Let's check its parsing branch:")
        print(f"    isinstance(item, dict) -> {isinstance(s0, dict)}")
        print(f"    hasattr(item, 'text') or hasattr(item, 'start') -> {hasattr(s0, 'text') or hasattr(s0, 'start')}")
        print(f"    hasattr(item, 'get') -> {hasattr(s0, 'get')}")
        if not isinstance(s0, dict) and not (hasattr(s0, 'text') or hasattr(s0, 'start')) and not hasattr(s0, 'get'):
            print(f"    RESULT: NONE OF THE BRANCHES MATCH -> snippet is SILENTLY SKIPPED")
        else:
            print(f"    RESULT: A branch matches, item will be parsed")
    print(f"\n  TRANSCRIPT SAMPLE (first 3 snippets):")
    for i, snip in enumerate(snippets[:3]):
        if hasattr(snip, 'text'):
            print(f"    [{i}] start={snip.start:.1f}s text={snip.text!r}")
        elif isinstance(snip, dict):
            print(f"    [{i}] start={snip.get('start',0):.1f}s text={snip.get('text','')!r}")
        else:
            print(f"    [{i}] {repr(snip)[:120]}")
    print(f"\n  TRANSCRIPT IS AVAILABLE: YES ({len(snippets)} entries)")
except Exception as e:
    print(f"  api.fetch() FAILED: {type(e).__name__}: {str(e)[:400]}")

# ─────────────────────────────────────────────────────────────────────
# CHECK 6: Run the actual YouTubeConnector.fetch_documents()
# ─────────────────────────────────────────────────────────────────────
print(f"\n{MINI}")
print("CHECK 6: FULL YouTubeConnector.fetch_documents() CALL")
print(MINI)

try:
    from connectors.youtube import YouTubeConnector
    connector = YouTubeConnector(urls=[TEST_URL])
    print(f"  Calling fetch_documents() on {TEST_URL}...")
    docs = connector.fetch_documents()
    print(f"  Documents returned: {len(docs)}")
    if docs:
        doc = docs[0]
        raw = doc.get("raw_data", b"")
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        print(f"  Source: {doc.get('source','')}")
        print(f"  Extension: {doc.get('extension','')}")
        print(f"  Metadata: {doc.get('metadata',{}).keys()}")
        print(f"  Transcript bytes: {len(raw)}")
        print(f"  Transcript preview:\n    {text[:400]!r}")
    else:
        print("  RESULT: fetch_documents() returned EMPTY LIST")
        print("  -> Both Tier 1 and Tier 2 failed silently")
except Exception as e:
    print(f"  fetch_documents() raised: {type(e).__name__}: {str(e)[:400]}")

# ─────────────────────────────────────────────────────────────────────
# CHECK 7: How the connector handles the v1.2.4 FetchedTranscript object
# ─────────────────────────────────────────────────────────────────────
print(f"\n{MINI}")
print("CHECK 7: CONNECTOR'S ITEM PARSING BRANCHES vs v1.2.4 FetchedTranscript")
print(MINI)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    result = api.fetch(video_id)
    items = list(result)
    if items:
        item = items[0]
        print(f"  Item type: {type(item).__name__}")
        print()
        print("  CONNECTOR PARSING BRANCH ANALYSIS (retriever.py lines 138-151):")
        print(f"  if isinstance(item, dict):                      -> {isinstance(item, dict)}")
        print(f"  elif hasattr(item,'text') or hasattr(item,'start'): -> {hasattr(item,'text') or hasattr(item,'start')}")
        print(f"  elif hasattr(item, 'get'):                      -> {hasattr(item, 'get')}")
        print(f"  else: continue (SILENTLY DROPPED)               -> {not isinstance(item, dict) and not (hasattr(item,'text') or hasattr(item,'start')) and not hasattr(item,'get')}")

        if hasattr(item, 'text') or hasattr(item, 'start'):
            text_val = str(getattr(item, "text", "")).strip()
            start_val = float(getattr(item, "start", 0.0))
            dur_val   = float(getattr(item, "duration", 0.0))
            print(f"\n  BRANCH 2 MATCH: getattr approach")
            print(f"    text     = {text_val!r}")
            print(f"    start    = {start_val}")
            print(f"    duration = {dur_val}")
            print(f"    Will snippet be appended? text truthy = {bool(text_val)}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("DIAGNOSTIC SUMMARY")
print(DIVIDER)
