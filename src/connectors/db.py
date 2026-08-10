import json
import sqlite3
from typing import List, Dict, Any, Tuple, Optional
from src.connectors.base import BaseConnector
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
# Relational Driver Wrappers
# ─────────────────────────────────────────────

class SQLiteDriver:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_tables(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

    def get_table_schema(self, table_name: str) -> Tuple[List[str], Optional[str]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            rows = cursor.fetchall()
            columns = [row["name"] for row in rows]
            pk = None
            for row in rows:
                if row["pk"] > 0:
                    pk = row["name"]
                    break
            return columns, pk

    def fetch_rows(self, table_name: str) -> List[List[Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM '{table_name}';")
            return [list(row) for row in cursor.fetchall()]


class MySQLDriver:
    def __init__(self, connection_params: Dict[str, Any]):
        self.params = connection_params

    def _get_connection(self) -> Any:
        import pymysql
        return pymysql.connect(
            host=self.params.get("host", "localhost"),
            user=self.params.get("user"),
            password=self.params.get("password"),
            database=self.params.get("database"),
            port=int(self.params.get("port", 3306)),
            charset='utf8mb4'
        )

    def get_tables(self) -> List[str]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES;")
                return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_table_schema(self, table_name: str) -> Tuple[List[str], Optional[str]]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"DESCRIBE `{table_name}`;")
                rows = cursor.fetchall()
                columns = [row[0] for row in rows]
                pk = None
                for row in rows:
                    if row[3] == "PRI":  # PRI stands for Primary Key in DESCRIBE query
                        pk = row[0]
                        break
                return columns, pk
        finally:
            conn.close()

    def fetch_rows(self, table_name: str) -> List[List[Any]]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT * FROM `{table_name}`;")
                return [list(row) for row in cursor.fetchall()]
        finally:
            conn.close()


class PostgreSQLDriver:
    def __init__(self, connection_params: Dict[str, Any]):
        self.params = connection_params

    def _get_connection(self) -> Any:
        import psycopg2
        return psycopg2.connect(
            host=self.params.get("host", "localhost"),
            user=self.params.get("user"),
            password=self.params.get("password"),
            database=self.params.get("database"),
            port=int(self.params.get("port", 5432))
        )

    def get_tables(self) -> List[str]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public';
                    """
                )
                return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_table_schema(self, table_name: str) -> Tuple[List[str], Optional[str]]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position;
                    """
                )
                columns = [row[0] for row in cursor.fetchall()]
                
                # Fetch primary key constraint name mapped to columns
                cursor.execute(
                    f"""
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                      AND tc.table_name = '{table_name}';
                    """
                )
                pk_rows = cursor.fetchall()
                pk = pk_rows[0][0] if pk_rows else None
                return columns, pk
        finally:
            conn.close()

    def fetch_rows(self, table_name: str) -> List[List[Any]]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f'SELECT * FROM "{table_name}";')
                return [list(row) for row in cursor.fetchall()]
        finally:
            conn.close()


# ─────────────────────────────────────────────
# Unified Database Ingestion Connector
# ─────────────────────────────────────────────

class DatabaseConnector(BaseConnector):
    def __init__(self, db_type: str, config: Dict[str, Any]):
        self.db_type = db_type.lower()
        self.config = config
        self.connector = None
        
        if self.db_type == "sqlite":
            self.connector = SQLiteDriver(config.get("db_path", ""))
        elif self.db_type == "mysql":
            self.connector = MySQLDriver(config)
        elif self.db_type == "postgresql":
            self.connector = PostgreSQLDriver(config)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def fetch_documents(self, selected_tables: List[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves table schemas and rows, packaging them for the structured ingestion parser.
        """
        docs = []
        try:
            tables = selected_tables if selected_tables else self.connector.get_tables()
        except Exception as e:
            logger.error(f"Failed to fetch table list from {self.db_type}: {e}")
            return []
        
        for table in tables:
            try:
                columns, pk = self.connector.get_table_schema(table)
                rows = self.connector.fetch_rows(table)
                
                # Package database information as a structured dict
                payload = {
                    "table_name": table,
                    "primary_key_col": pk,
                    "columns": columns,
                    "rows": rows
                }
                
                # Encode as json bytes
                raw_data = json.dumps(payload).encode("utf-8")
                
                # Construct connection URL for citations
                source_url = f"{self.db_type}://{self.config.get('host', 'localhost')}/{self.config.get('database', '')}/{table}"
                if self.db_type == "sqlite":
                    source_url = f"sqlite:///{self.config.get('db_path')}/{table}"
                
                meta = {
                    "source_type": "database",
                    "db_type": self.db_type,
                    "source": source_url,
                    "table_name": table
                }
                
                docs.append({
                    "raw_data": raw_data,
                    "source": source_url,
                    "extension": ".json",  # Denotes DB payload to ParserFactory
                    "metadata": meta
                })
            except Exception as e:
                logger.error(f"Failed to fetch table {table} from {self.db_type}: {e}")
                
        return docs
