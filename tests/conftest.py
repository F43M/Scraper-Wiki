import sys
from types import SimpleNamespace, ModuleType


class _DummyMetric:
    def __init__(self, *a, **k):
        pass

    def inc(self, *a, **k):
        pass

    def observe(self, *a, **k):
        pass

    def set(self, *a, **k):
        pass


prometheus_stub = SimpleNamespace(
    Counter=lambda *a, **k: _DummyMetric(),
    Histogram=lambda *a, **k: _DummyMetric(),
    Gauge=lambda *a, **k: _DummyMetric(),
    start_http_server=lambda *a, **k: None,
)
sys.modules.setdefault("prometheus_client", prometheus_stub)

pyarrow_stub = ModuleType("pyarrow")
pyarrow_stub.Table = object
pyarrow_stub.parquet = SimpleNamespace(write_table=lambda *a, **k: None)
pyarrow_stub.ipc = SimpleNamespace()
pyarrow_stub.csv = SimpleNamespace(read_csv=lambda *a, **k: None)
sys.modules.setdefault("pyarrow", pyarrow_stub)
sys.modules.setdefault("pyarrow.parquet", pyarrow_stub.parquet)
sys.modules.setdefault("pyarrow.ipc", pyarrow_stub.ipc)
sys.modules.setdefault("pyarrow.csv", pyarrow_stub.csv)

sys.modules.setdefault("yaml", SimpleNamespace(safe_load=lambda *a, **k: {}))
