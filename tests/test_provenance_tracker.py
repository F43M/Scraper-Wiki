import tempfile
from pathlib import Path

from provenance import tracker


def test_record_and_fetch():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "prov.sqlite"
        meta = tracker.record_provenance("http://x", "content", db_path=str(db))
        assert "retrieved_at" in meta and "content_hash" in meta
        fetched = tracker.get_provenance("http://x", db_path=str(db))
        assert fetched == meta


def test_should_fetch_logic():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "prov.sqlite"
        url = "http://y"
        assert tracker.should_fetch(url, db_path=str(db))
        tracker.record_provenance(url, "c", db_path=str(db))
        assert not tracker.should_fetch(url, db_path=str(db))
        assert tracker.should_fetch(url, max_age_hours=0, db_path=str(db))
