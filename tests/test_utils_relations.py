import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Dummy NLP components
class DummyToken:
    def __init__(self, text, dep, i=0, lemma=None):
        self.text = text
        self.dep_ = dep
        self.i = i
        self.lemma_ = lemma or text
        self.children = []

class DummyEnt:
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end
        self.label_ = "PERSON"

class DummySent(list):
    pass

class DummyDoc:
    def __init__(self):
        sub = DummyToken("Guido", "nsubj", i=0)
        root = DummyToken("created", "ROOT", i=1, lemma="create")
        obj = DummyToken("Python", "dobj", i=2)
        root.children = [sub, obj]
        self.tokens = [sub, root, obj]
        self.sents = [DummySent(self.tokens)]
        self.ents = [DummyEnt("Guido", 0, 1), DummyEnt("Python", 2, 3)]

    def __iter__(self):
        return iter(self.tokens)

class DummyNLP:
    def __call__(self, text):
        return DummyDoc()

class DummyProc:
    @classmethod
    def get_instance(cls, lang):
        return DummyNLP()


def test_extract_relations_simple(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=lambda *a, **k: DummyNLP()))
    monkeypatch.setitem(sys.modules, "scraper_wiki", SimpleNamespace(NLPProcessor=DummyProc))

    relations_mod = importlib.reload(importlib.import_module("utils.relation"))
    rels = relations_mod.extract_relations("Guido created Python", "en")
    assert rels == [{"subject": "Guido", "relation": "create", "object": "Python"}]
