from __future__ import annotations
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    system TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
"""


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def new_conversation(self, model: str, system: str | None) -> int:
        cur = self.conn.execute(
            "INSERT INTO conversations (model, system, created_at) VALUES (?, ?, ?)",
            (model, system, time.time()),
        )
        cid = cur.lastrowid
        self.conn.commit()
        if system:
            self.append_message(cid, "system", system)
        return cid

    def append_message(self, conversation_id: int, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, time.time()),
        )
        self.conn.commit()

    def load_messages(self, conversation_id: int) -> list[dict]:
        cur = self.conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
        return [{"role": r, "content": c} for (r, c) in cur.fetchall()]

    def last_conversation_id(self) -> int | None:
        cur = self.conn.execute("SELECT id FROM conversations ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else None

    def list_conversations(self, limit: int = 20) -> list[tuple]:
        cur = self.conn.execute(
            "SELECT c.id, c.model, c.created_at, "
            "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id AND m.role != 'system') "
            "FROM conversations c ORDER BY c.id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

    def close(self) -> None:
        self.conn.close()
