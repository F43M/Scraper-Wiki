import importlib
import sys
from types import SimpleNamespace, ModuleType
from pathlib import Path

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies to avoid installation
sys.modules['sentence_transformers'] = SimpleNamespace(SentenceTransformer=object)
sys.modules.setdefault('datasets', SimpleNamespace(Dataset=object, concatenate_datasets=lambda *a, **k: None))
sys.modules.setdefault('spacy', SimpleNamespace(load=lambda *a, **k: None))
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

# Provide missing attribute in wikipediaapi
import wikipediaapi
if not hasattr(wikipediaapi, 'WikipediaException'):
    wikipediaapi.WikipediaException = Exception

import scraper_wiki as sw


class DummyEmbed:
    def encode(self, *args, **kwargs):
        import numpy as np
        return np.zeros((len(args[0]) if args else 1, 3))


sw.NLPProcessor.get_embedding_model = classmethod(lambda cls: DummyEmbed())

class DummyNLP:
    def __call__(self, text):
        class Doc:
            noun_chunks = [SimpleNamespace(text='Python')]
            ents = []
            sents = []
        return Doc()

sw.NLPProcessor.get_instance = classmethod(lambda cls, lang: DummyNLP())


def test_translate_question_pt():
    qb = sw.DatasetBuilder()
    question = qb._translate_question('What is Python?', 'pt')
    assert 'O que é' in question

def test_translate_question_es():
    qb = sw.DatasetBuilder()
    question = qb._translate_question('How does Python relate to programacion?', 'es')
    assert 'Cómo' in question and 'se relaciona con' in question

def test_translate_question_fr():
    qb = sw.DatasetBuilder()
    question = qb._translate_question('How does Python relate to programmation?', 'fr')
    assert 'Comment' in question and 'se rapporte à' in question

def test_classify_topic_ai_nlp():
    qb = sw.DatasetBuilder()
    topic, sub = qb._classify_topic('Deep Learning', 'O processamento de linguagem natural evoluiu', 'pt')
    assert topic == 'ai'
    assert sub == 'nlp'

def test_advanced_clean_text_removes_html():
    raw = '<div>Example</div>\n[[1]]\n== Referências ==\nTexto'
    cleaned = sw.advanced_clean_text(raw, 'pt')
    assert '<div>' not in cleaned and 'Referências' not in cleaned

def test_extract_main_content():
    html = '<html><body><div id="mw-content-text">content</div><table></table></body></html>'
    result = sw.extract_main_content(html)
    assert 'content' in result and 'table' not in result


def test_nlp_triple_fallback(monkeypatch):
    import importlib

    sw_reloaded = importlib.reload(sw)
    sw_reloaded.NLPProcessor._instances = {}
    sw_reloaded.NLPProcessor.get_embedding_model = classmethod(lambda cls: DummyEmbed())

    calls = []

    def fake_load(name, *args, **kwargs):
        calls.append(name)
        if len(calls) < 3:
            raise OSError('missing')
        return DummyNLP()

    monkeypatch.setattr(sw_reloaded.spacy, 'load', fake_load)

    nlp = sw_reloaded.NLPProcessor.get_instance('pt')
    assert isinstance(nlp, DummyNLP)
    assert calls == ['pt_core_news_lg', 'pt_core_news_sm', 'en_core_web_sm']
