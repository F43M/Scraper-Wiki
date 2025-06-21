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


def test_extract_relations_no_match(monkeypatch):
    class EmptyDoc:
        def __init__(self):
            root = DummyToken('is', 'ROOT', i=0, lemma='be')
            self.tokens = [root]
            self.sents = [DummySent([root])]
            self.ents = []
        def __iter__(self):
            return iter(self.tokens)

    class EmptyNLP:
        def __call__(self, text):
            return EmptyDoc()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=lambda *a, **k: EmptyNLP()))
    monkeypatch.setitem(sys.modules, "scraper_wiki", SimpleNamespace(NLPProcessor=DummyProc))

    relations_mod = importlib.reload(importlib.import_module("utils.relation"))
    rels = relations_mod.extract_relations("Just words", "en")
    assert rels == []


def test_extract_relations_multiple(monkeypatch):
    class MultiDoc:
        def __init__(self):
            s1_sub = DummyToken('Guido', 'nsubj', i=0)
            s1_root = DummyToken('created', 'ROOT', i=1, lemma='create')
            s1_obj = DummyToken('Python', 'dobj', i=2)
            s1_root.children = [s1_sub, s1_obj]
            s2_sub = DummyToken('Python', 'nsubj', i=3)
            s2_root = DummyToken('powers', 'ROOT', i=4, lemma='power')
            s2_obj = DummyToken('programs', 'dobj', i=5)
            s2_root.children = [s2_sub, s2_obj]
            self.tokens = [s1_sub, s1_root, s1_obj, s2_sub, s2_root, s2_obj]
            self.sents = [DummySent(self.tokens[:3]), DummySent(self.tokens[3:])]
            self.ents = [DummyEnt('Guido', 0, 1), DummyEnt('Python', 2, 3)]
        def __iter__(self):
            return iter(self.tokens)

    class MultiNLP:
        def __call__(self, text):
            return MultiDoc()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=lambda *a, **k: MultiNLP()))
    monkeypatch.setitem(sys.modules, "scraper_wiki", SimpleNamespace(NLPProcessor=DummyProc))

    relations_mod = importlib.reload(importlib.import_module("utils.relation"))
    rels = relations_mod.extract_relations("Two sentences", "en")
    assert rels == [
        {"subject": "Guido", "relation": "create", "object": "Python"},
        {"subject": "Python", "relation": "power", "object": "programs"},
    ]
