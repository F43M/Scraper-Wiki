import os
import json
import sqlite3
from typing import Optional


def save_to_db(
    data: dict, table: str = "infoboxes", db_path: str = "infoboxes.sqlite"
) -> None:
    """Save a dictionary as a JSON blob in the given SQLite table."""
    dir_name = os.path.dirname(db_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT)"
        )
        conn.execute(
            f"INSERT INTO {table} (data) VALUES (?)",
            (json.dumps(data, ensure_ascii=False),),
        )
        conn.commit()
    finally:
        conn.close()


def get_last_processed(name: str, db_path: str = "metadata.sqlite") -> Optional[str]:
    """Return the last processed timestamp stored for ``name``."""
    if os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, ts TEXT)"
        )
        row = conn.execute("SELECT ts FROM metadata WHERE name=?", (name,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_last_processed(
    name: str, timestamp: str, db_path: str = "metadata.sqlite"
) -> None:
    """Update the last processed timestamp for ``name``."""
    if os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, ts TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata (name, ts) VALUES (?, ?)",
            (name, timestamp),
        )
        conn.commit()
    finally:
        conn.close()
