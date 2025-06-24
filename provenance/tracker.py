import os
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict


def _ensure_db(db_path: str) -> sqlite3.Connection:
    dir_name = os.path.dirname(db_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS provenance (url TEXT PRIMARY KEY, ts TEXT, hash TEXT)"
    )
    return conn


def record_provenance(
    url: str, content: str, db_path: str = "provenance.sqlite"
) -> Dict[str, str]:
    ts = datetime.utcnow().isoformat()
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    conn = _ensure_db(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO provenance (url, ts, hash) VALUES (?, ?, ?)",
                (url, ts, h),
            )
    finally:
        conn.close()
    return {"retrieved_at": ts, "content_hash": h}


def get_provenance(
    url: str, db_path: str = "provenance.sqlite"
) -> Optional[Dict[str, str]]:
    conn = _ensure_db(db_path)
    try:
        row = conn.execute(
            "SELECT ts, hash FROM provenance WHERE url=?", (url,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    ts, h = row
    return {"retrieved_at": ts, "content_hash": h}


def should_fetch(
    url: str, max_age_hours: int = 24, db_path: str = "provenance.sqlite"
) -> bool:
    info = get_provenance(url, db_path)
    if not info:
        return True
    ts = datetime.fromisoformat(info["retrieved_at"])
    if datetime.utcnow() - ts > timedelta(hours=max_age_hours):
        return True
    return False
