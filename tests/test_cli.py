import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies to avoid installation
sys.modules.setdefault('sentence_transformers', SimpleNamespace(SentenceTransformer=object))
sys.modules.setdefault('datasets', SimpleNamespace(Dataset=object, concatenate_datasets=lambda *a, **k: None))
sys.modules.setdefault('spacy', SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault('unidecode', SimpleNamespace(unidecode=lambda x: x))
sys.modules.setdefault('tqdm', SimpleNamespace(tqdm=lambda x, **k: x))
sys.modules.setdefault('html2text', SimpleNamespace())
sys.modules.setdefault('backoff', SimpleNamespace(on_exception=lambda *a, **k: (lambda f: f), expo=lambda *a, **k: None))

sklearn_mod = ModuleType('sklearn')
sklearn_mod.cluster = SimpleNamespace(KMeans=object)
sklearn_mod.feature_extraction = SimpleNamespace(text=SimpleNamespace(TfidfVectorizer=object))
sys.modules.setdefault('sklearn', sklearn_mod)
sys.modules.setdefault('sklearn.cluster', sklearn_mod.cluster)
sys.modules.setdefault('sklearn.feature_extraction', sklearn_mod.feature_extraction)
sys.modules.setdefault('sklearn.feature_extraction.text', sklearn_mod.feature_extraction.text)

sumy_mod = ModuleType('sumy')
parsers_mod = ModuleType('sumy.parsers')
plaintext_mod = ModuleType('sumy.parsers.plaintext')
plaintext_mod.PlaintextParser = object
parsers_mod.plaintext = plaintext_mod
nlp_mod = ModuleType('sumy.nlp')
tokenizers_mod = ModuleType('sumy.nlp.tokenizers')
tokenizers_mod.Tokenizer = object
nlp_mod.tokenizers = tokenizers_mod
summarizers_mod = ModuleType('sumy.summarizers')
lsa_mod = ModuleType('sumy.summarizers.lsa')
lsa_mod.LsaSummarizer = object
summarizers_mod.lsa = lsa_mod
sumy_mod.parsers = parsers_mod
sumy_mod.nlp = nlp_mod
sumy_mod.summarizers = summarizers_mod
sys.modules.setdefault('sumy', sumy_mod)
sys.modules.setdefault('sumy.parsers', parsers_mod)
sys.modules.setdefault('sumy.parsers.plaintext', plaintext_mod)
sys.modules.setdefault('sumy.nlp', nlp_mod)
sys.modules.setdefault('sumy.nlp.tokenizers', tokenizers_mod)
sys.modules.setdefault('sumy.summarizers', summarizers_mod)
sys.modules.setdefault('sumy.summarizers.lsa', lsa_mod)
sys.modules.setdefault('streamlit', SimpleNamespace())
sys.modules.setdefault('psutil', SimpleNamespace(cpu_percent=lambda interval=1: 0, virtual_memory=lambda: SimpleNamespace(percent=0)))

import wikipediaapi
if not hasattr(wikipediaapi, 'WikipediaException'):
    wikipediaapi.WikipediaException = Exception

from typer.testing import CliRunner
import cli


def test_status_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.scraper_wiki.Config, "OUTPUT_DIR", str(tmp_path))
    (tmp_path / "example.json").write_text("{}")

    runner = CliRunner()
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "example.json" in result.output
    assert str(tmp_path) in result.output
