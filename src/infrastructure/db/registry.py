import sqlite3
import time
from typing import Optional, Tuple
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel

class RegistryRecord(BaseModel):
    logical_id: str
    filename: str
    current_version: int
    content_hash: str
    updated_at: float

class DocumentRegistry:
    def __init__(self, db_path: str = "data/registry.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                logical_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL UNIQUE,
                current_version INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def get_by_filename(self, filename: str) -> Optional[RegistryRecord]:
        """Fetch document state by filename."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return RegistryRecord(
                logical_id=row[0],
                filename=row[1],
                current_version=row[2],
                content_hash=row[3],
                updated_at=row[4]
            )
        return None

    def upsert_document(self, filename: str, content_hash: str, logical_id: Optional[str] = None, version: int = 1) -> str:
        """
        Register a new document or update an existing one.
        Returns logical_id.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = time.time()
        
        # Check if exists (handled by caller usually, but safe upsert here)
        existing = self.get_by_filename(filename)
        
        if existing:
            # Update
            lid = existing.logical_id
            cursor.execute("""
                UPDATE documents 
                SET current_version = ?, content_hash = ?, updated_at = ?
                WHERE logical_id = ?
            """, (version, content_hash, timestamp, lid))
        else:
            # Insert
            lid = logical_id or str(uuid4())
            cursor.execute("""
                INSERT INTO documents (logical_id, filename, current_version, content_hash, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (lid, filename, version, content_hash, timestamp))
            
        conn.commit()
        conn.close()
        return lid
