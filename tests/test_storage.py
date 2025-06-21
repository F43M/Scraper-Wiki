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



def test_s3_save_tfrecord(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.kwargs = None
        def put_object(self, Bucket, Key, Body):
            self.kwargs = {'Bucket': Bucket, 'Key': Key, 'Body': Body}

    dummy = DummyClient()
    monkeypatch.setitem(sys.modules, 'boto3', SimpleNamespace(client=lambda *a, **k: dummy))
    import importlib
    storage_mod = importlib.reload(importlib.import_module('storage'))
    s3 = storage_mod.S3Storage('bucket', prefix='pre', client=dummy)
    data = [{'x': 1}]
    s3.save_dataset(data, fmt='tfrecord')
    assert dummy.kwargs['Key'] == 'pre/wikipedia_qa.tfrecord'
    assert b'{"x": 1}' in dummy.kwargs['Body']
