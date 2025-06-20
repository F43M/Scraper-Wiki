import importlib
import sys
from types import SimpleNamespace

# ensure repo root on path
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class DummyEnt:
    def __init__(self, text, label):
        self.text = text
        self.label_ = label

class DummyDoc:
    def __init__(self):
        self.ents = [DummyEnt('Python', 'ORG'), DummyEnt('Guido', 'PERSON')]

class DummyNLP:
    def __call__(self, text):
        return DummyDoc()

sys.modules['spacy'] = SimpleNamespace(load=lambda *a, **k: DummyNLP())
text_mod = importlib.import_module('utils.text')
sys.modules['spacy'] = SimpleNamespace(load=lambda *a, **k: None)


def test_clean_text_basic():
    assert text_mod.clean_text('A [1]\nB  C') == 'A B C'


def test_normalize_person():
    data = {'name': 'Guido', 'birth_date': '1956', 'occupation': 'Programmer|BDFL'}
    result = text_mod.normalize_person(data)
    assert result['name'] == 'Guido'
    assert result['occupation'] == ['Programmer', 'BDFL']


def test_extract_entities(monkeypatch):
    monkeypatch.setattr(text_mod, 'nlp', DummyNLP())
    ents = text_mod.extract_entities('Any text')
    assert {'text': 'Python', 'type': 'ORG'} in ents
    assert {'text': 'Guido', 'type': 'PERSON'} in ents
