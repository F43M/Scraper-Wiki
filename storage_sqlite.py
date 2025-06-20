import os
import json
import sqlite3


def save_to_db(data: dict, table: str = "infoboxes", db_path: str = "infoboxes.sqlite") -> None:
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
