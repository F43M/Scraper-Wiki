import json
import sqlite3
from storage_sqlite import save_to_db


def test_save_to_db_creates_table_and_inserts(tmp_path):
    db_file = tmp_path / "test.sqlite"
    data = {"a": 1}
    save_to_db(data, table="info", db_path=str(db_file))
    conn = sqlite3.connect(db_file)
    row = conn.execute("SELECT data FROM info").fetchone()
    conn.close()
    assert row is not None
    assert json.loads(row[0]) == data
