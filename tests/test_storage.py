import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub tensorflow to avoid heavy dependency
class DummyWriter:
    def __init__(self, path):
        self.f = open(path, 'wb')
    def write(self, data):
        self.f.write(data)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.f.close()

sys.modules['tensorflow'] = SimpleNamespace(io=SimpleNamespace(TFRecordWriter=DummyWriter))

from storage import LocalStorage


def test_save_tfrecord(tmp_path):
    storage = LocalStorage(str(tmp_path))
    data = [{"a": 1}, {"b": 2}]
    storage.save_dataset(data, fmt='tfrecord')
    tf_file = tmp_path / 'wikipedia_qa.tfrecord'
    assert tf_file.exists()
    assert b'{"a": 1}' in tf_file.read_bytes()

