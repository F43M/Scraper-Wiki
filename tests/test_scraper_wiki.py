import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub sentence_transformers to avoid heavy dependency
sys.modules['sentence_transformers'] = SimpleNamespace(SentenceTransformer=object)

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
