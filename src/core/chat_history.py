import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
# Database Path
# ─────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "chat_history.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set for dict-like access."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create the chat_history table if it doesn't exist.
    Called once on application startup.
    """
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT    NOT NULL,
                session_date TEXT    NOT NULL,
                timestamp    TEXT    NOT NULL,
                role         TEXT    NOT NULL,
                content      TEXT    NOT NULL,
                confidence   REAL    DEFAULT 0.0,
                sources      TEXT    DEFAULT '[]'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT    NOT NULL,
                filename     TEXT    NOT NULL,
                upload_date  TEXT    NOT NULL,
                chunk_count  INTEGER NOT NULL,
                UNIQUE(user_id, filename)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT    NOT NULL,
                query_text   TEXT    NOT NULL,
                timestamp    TEXT    NOT NULL,
                latency      REAL    NOT NULL,
                confidence   REAL    NOT NULL,
                is_answered  INTEGER NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT    NOT NULL,
                query_text   TEXT,
                rating       TEXT    NOT NULL,
                timestamp    TEXT    NOT NULL
            )
        """)
        
        # Migrate/Alter table to add new columns if they do not exist
        cursor = conn.execute("PRAGMA table_info(documents)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "drive_file_id" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN drive_file_id TEXT")
        if "version" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN version TEXT")
        if "modified_time" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN modified_time TEXT")
            
        cursor = conn.execute("PRAGMA table_info(chat_history)")
        chat_columns = [row["name"] for row in cursor.fetchall()]
        if "session_id" not in chat_columns:
            conn.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT DEFAULT 'default'")
            
        conn.commit()
    logger.info("Chat history database initialized.")


def save_feedback(user_id: str, query: str, rating: str):
    """Save user feedback rating ('thumbs_up' or 'thumbs_down') for a query response."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback (user_id, query_text, rating, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, query, rating, now)
            )
            conn.commit()
        logger.info(f"Saved feedback [{rating}] for user {user_id}.")
    except Exception as e:
        logger.error(f"Failed to save user feedback: {e}")


def save_chat(
    user_id: str,
    role: str,
    content: str,
    confidence: float = 0.0,
    sources: Optional[List[Dict[str, Any]]] = None,
    session_id: str = 'default'
):
    """
    Save a single chat message (user or assistant) to the SQLite database.

    Args:
        role:       'user' or 'assistant'
        content:    The message text
        confidence: Retrieval confidence score (for assistant messages)
        sources:    List of source chunk dicts (for assistant messages)
        session_id: Optional unique identifier for the conversation session
    """
    now = datetime.now()
    session_date = now.strftime("%Y-%m-%d")
    timestamp    = now.strftime("%H:%M:%S")

    # Serialize sources — keep metadata, YouTube attributes, and text snippet
    serializable_sources = []
    for src in (sources or []):
        meta = src.get("metadata", {}) if isinstance(src.get("metadata"), dict) else {}
        s_name = meta.get("source") or src.get("source") or "Unknown"
        p_num = meta.get("page_number") or src.get("page_number") or "N/A"
        snip = src.get("text") or src.get("snippet") or ""
        s_type = meta.get("source_type") or src.get("source_type") or ("youtube" if "youtube" in str(s_name).lower() else "file")
        v_url = meta.get("video_url_timestamped") or src.get("video_url_timestamped") or ""
        v_title = meta.get("video_title") or src.get("video_title") or ""
        t_range = meta.get("formatted_time_range") or src.get("formatted_time_range") or ""
        s_fmt = meta.get("start_formatted") or src.get("start_formatted") or ""

        serializable_sources.append({
            "source": s_name,
            "page_number": p_num,
            "snippet": snip[:300],
            "text": snip[:300],
            "rerank_score": float(src.get("rerank_score", 0.0)),
            "source_type": s_type,
            "video_url_timestamped": v_url,
            "video_title": v_title,
            "formatted_time_range": t_range,
            "start_formatted": s_fmt
        })

    try:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_history (user_id, session_date, timestamp, role, content, confidence, sources, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, session_date, timestamp, role, content, confidence, json.dumps(serializable_sources), session_id)
            )
            conn.commit()
        logger.info(f"Saved [{role}] message for user {user_id} in session {session_id}.")
    except Exception as e:
        logger.error(f"Failed to save chat message: {e}")


def load_session_history(user_id: str, session_id: str = 'default', limit: int = 10) -> List[Dict[str, Any]]:
    """
    Load the last N messages for a specific session.
    """
    try:
        with _get_connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content, confidence, sources, timestamp
                FROM chat_history
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, session_id, limit)
            ).fetchall()
            
        messages = []
        for row in reversed(rows):
            messages.append({
                "role": row["role"],
                "content": row["content"],
                "confidence": row["confidence"],
                "sources": json.loads(row["sources"]) if row["sources"] else [],
                "timestamp": row["timestamp"]
            })
        return messages
    except Exception as e:
        logger.error(f"Failed to load session history for {session_id}: {e}")
        return []


def load_chat_history(user_id: str, limit_days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load chat messages for a specific user grouped by date.
    """
    try:
        with _get_connection() as conn:
            rows = conn.execute(
                """
                SELECT session_date, timestamp, role, content, confidence, sources
                FROM chat_history
                WHERE user_id = ? AND session_date >= date('now', ?)
                ORDER BY session_date DESC, id ASC
                """,
                (user_id, f"-{limit_days} days")
            ).fetchall()

        # Group by date
        history: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            date = row["session_date"]
            if date not in history:
                history[date] = []
            history[date].append({
                "timestamp":  row["timestamp"],
                "role":       row["role"],
                "content":    row["content"],
                "confidence": row["confidence"],
                "sources":    json.loads(row["sources"])
            })

        logger.info(f"Loaded chat history: {len(history)} day(s), {len(rows)} total messages.")
        return history

    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
        return {}


def load_today_history(user_id: str) -> List[Dict[str, Any]]:
    """
    Load only today's messages for a specific user.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    full_history = load_chat_history(user_id, limit_days=1)
    return full_history.get(today, [])


def load_messages_for_date(user_id: str, date_str: str) -> List[Dict[str, Any]]:
    """
    Load all messages for a specific date and user.
    """
    try:
        with _get_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, role, content, confidence, sources
                FROM chat_history
                WHERE user_id = ? AND session_date = ?
                ORDER BY id ASC
                """,
                (user_id, date_str)
            ).fetchall()

        return [{
            "timestamp":  row["timestamp"],
            "role":       row["role"],
            "content":    row["content"],
            "confidence": row["confidence"],
            "sources":    json.loads(row["sources"])
        } for row in rows]

    except Exception as e:
        logger.error(f"Failed to load messages for {date_str}: {e}")
        return []


def clear_history(user_id: str):
    """Delete all chat history for a specific user."""
    try:
        with _get_connection() as conn:
            conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
            conn.commit()
        logger.info(f"Chat history cleared for user {user_id}.")
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")


# ─────────────────────────────────────────────
# Document Metadata Functions
# ─────────────────────────────────────────────

def save_document_meta(
    user_id: str,
    filename: str,
    chunk_count: int,
    drive_file_id: Optional[str] = None,
    version: Optional[str] = None,
    modified_time: Optional[str] = None
):
    """Save document metadata for a specific user (supports Google Drive metadata)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents 
                (user_id, filename, upload_date, chunk_count, drive_file_id, version, modified_time) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, filename, now, chunk_count, drive_file_id, version, modified_time)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to save document metadata: {e}")


def load_document_meta(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load document metadata (if user_id is provided, filter by user; if None or 'all', load all)."""
    try:
        with _get_connection() as conn:
            if not user_id or user_id == "all":
                rows = conn.execute(
                    "SELECT filename, upload_date, chunk_count, drive_file_id, version, modified_time, user_id FROM documents ORDER BY upload_date DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT filename, upload_date, chunk_count, drive_file_id, version, modified_time, user_id FROM documents WHERE user_id = ? ORDER BY upload_date DESC",
                    (user_id,)
                ).fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to load document metadata: {e}")
        return []


def delete_document_meta(user_id: str, filename: str):
    """Delete document metadata for a specific user."""
    try:
        with _get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE user_id = ? AND filename = ?", (user_id, filename))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to delete document metadata: {e}")


def clear_all_metadata(user_id: str):
    """Clear all metadata for a specific user."""
    try:
        with _get_connection() as conn:
            conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM queries WHERE user_id = ?", (user_id,))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to clear user metadata: {e}")


# ─────────────────────────────────────────────
# Analytics Functions
# ─────────────────────────────────────────────

def save_query_metrics(user_id: str, query: str, latency: float, confidence: float, is_answered: bool):
    """Save performance metrics for a specific query."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT INTO queries (user_id, query_text, timestamp, latency, confidence, is_answered)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, query, now, latency, confidence, 1 if is_answered else 0)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to save query metrics: {e}")


def load_analytics_summary(user_id: str) -> Dict[str, Any]:
    """Load persistent analytics summary for a specific user."""
    try:
        with _get_connection() as conn:
            # Basic stats
            stats = conn.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    SUM(is_answered) as answered,
                    AVG(latency) as avg_lat,
                    AVG(confidence) as avg_conf
                FROM queries WHERE user_id = ?
                """,
                (user_id,)
            ).fetchone()
            
            # Confidence history
            conf_rows = conn.execute(
                "SELECT confidence FROM queries WHERE user_id = ? ORDER BY id ASC",
                (user_id,)
            ).fetchall()
            
            # Latency history
            lat_rows = conn.execute(
                "SELECT latency FROM queries WHERE user_id = ? ORDER BY id ASC",
                (user_id,)
            ).fetchall()

            # Query frequency
            freq_rows = conn.execute(
                "SELECT query_text, COUNT(*) as count FROM queries WHERE user_id = ? GROUP BY query_text ORDER BY count DESC LIMIT 10",
                (user_id,)
            ).fetchall()

            # Document usage from chat_history sources
            doc_counts = {}
            chat_rows = conn.execute("SELECT sources FROM chat_history WHERE user_id = ?", (user_id,)).fetchall()
            for row in chat_rows:
                if row["sources"]:
                    try:
                        sources = json.loads(row["sources"])
                        for s in sources:
                            name = s.get("source", "Unknown")
                            doc_counts[name] = doc_counts.get(name, 0) + 1
                    except:
                        continue

            return {
                "total_queries": stats["total"] or 0,
                "answered": stats["answered"] or 0,
                "rejected": (stats["total"] or 0) - (stats["answered"] or 0),
                "avg_confidence": round(stats["avg_conf"] or 0.0, 3),
                "avg_latency": round(stats["avg_lat"] or 0.0, 2),
                "confidence_history": [r["confidence"] for r in conf_rows],
                "response_times": [r["latency"] for r in lat_rows],
                "query_text_history": {r["query_text"]: r["count"] for r in freq_rows},
                "doc_query_count": doc_counts
            }
    except Exception as e:
        logger.error(f"Failed to load analytics summary: {e}")
        return {
            "total_queries": 0, "answered": 0, "rejected": 0,
            "avg_confidence": 0.0, "avg_latency": 0.0,
            "confidence_history": [], "response_times": [], "query_text_history": {}
        }


# Initialize DB on module import
init_db()
