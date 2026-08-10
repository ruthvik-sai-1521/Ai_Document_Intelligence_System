import sqlite3
import hashlib
import os
import uuid
import jwt
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from core.logger import setup_logger

logger = setup_logger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "chat_history.db"
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "documind_enterprise_secure_jwt_secret_key_2026")
JWT_ALGORITHM = "HS256"

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Generates a SHA-256 password hash using a 16-byte random salt."""
    if not salt:
        salt = os.urandom(16).hex()
    hashed = hashlib.sha256((password + salt).encode("utf-8")).hexdigest()
    return hashed, salt

def init_auth_db():
    """Create the users table if not existing and seed default admin and user accounts."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT    UNIQUE NOT NULL,
                username      TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                salt          TEXT    NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'user',
                created_at    TEXT    NOT NULL
            )
        """)
        conn.commit()
        
        # Check if default accounts exist, if not seed them
        cur = conn.execute("SELECT COUNT(*) as count FROM users")
        count = cur.fetchone()["count"]
        if count == 0:
            logger.info("Seeding default auth accounts: 'admin' and 'user'...")
            _seed_user(conn, "admin", "admin123", "admin")
            _seed_user(conn, "user", "user123", "user")
            conn.commit()

def _seed_user(conn: sqlite3.Connection, username: str, password: str, role: str):
    user_id = str(uuid.uuid4())
    pass_hash, salt = hash_password(password)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO users (user_id, username, password_hash, salt, role, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, username, pass_hash, salt, role, now)
    )

def register_user(username: str, password: str, role: str = "user") -> Tuple[bool, str]:
    """Register a new user account."""
    if not username.strip() or not password.strip():
        return False, "Username and password cannot be empty."
    
    user_id = str(uuid.uuid4())
    pass_hash, salt = hash_password(password)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, username, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, username.strip(), pass_hash, salt, role, now)
            )
            conn.commit()
        logger.info(f"Registered new user '{username}' with role '{role}'.")
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists."
    except Exception as e:
        logger.error(f"Failed to register user: {e}")
        return False, f"Registration error: {e}"

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate username and password against SQLite store."""
    try:
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT user_id, username, password_hash, salt, role FROM users WHERE username = ?",
                (username.strip(),)
            ).fetchone()
            
            if not row:
                return None
                
            expected_hash, _ = hash_password(password, row["salt"])
            if expected_hash == row["password_hash"]:
                return {
                    "user_id": row["user_id"],
                    "username": row["username"],
                    "role": row["role"]
                }
            return None
    except Exception as e:
        logger.error(f"Authentication error for {username}: {e}")
        return None

def create_access_token(user_data: Dict[str, Any], expires_delta_hours: int = 24) -> str:
    """Encodes a JWT access token containing user payload and expiration time."""
    payload = user_data.copy()
    expire = datetime.utcnow() + timedelta(hours=expires_delta_hours)
    payload.update({"exp": expire})
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired.")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return None

# Initialize auth DB on module load
init_auth_db()
