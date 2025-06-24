import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Load the module without executing ``training.__init__``
training_mod = ModuleType("training")
training_mod.__path__ = [str(ROOT / "training")]
sys.modules["training"] = training_mod
spec = importlib.util.spec_from_file_location(
    "training.airflow_pipeline", ROOT / "training" / "airflow_pipeline.py"
)
pipe = importlib.util.module_from_spec(spec)
sys.modules["training.airflow_pipeline"] = pipe
spec.loader.exec_module(pipe)


def test_create_dag():
    if pipe.DAG is None:
        with pytest.raises(RuntimeError):
            pipe.create_dag()
    else:
        dag = pipe.create_dag()
        assert {
            "collect_data",
            "clean_data",
            "fine_tune_model",
        }.issubset(dag.task_dict)
