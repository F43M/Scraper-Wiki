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


def test_fetch_html_content_success(monkeypatch):
    class DummyResp:
        def __init__(self):
            self.text = '<html></html>'
        def raise_for_status(self):
            pass
    def fake_get(*a, **k):
        return DummyResp()
    monkeypatch.setattr(sw.requests, 'get', fake_get)
    result = sw.fetch_html_content('Any', 'en')
    assert result == '<html></html>'


def test_fetch_html_content_error(monkeypatch):
    fake_cache = {}
    monkeypatch.setattr(sw, 'cache', SimpleNamespace(
        get=lambda k: fake_cache.get(k),
        set=lambda k, v: fake_cache.__setitem__(k, v),
        stats=lambda: {}
    ))

    def fake_get(*a, **k):
        raise sw.requests.exceptions.RequestException('fail')

    monkeypatch.setattr(sw.requests, 'get', fake_get)
    result = sw.fetch_html_content('Any', 'en')
    assert result == ''


class DummyFuture:
    def __init__(self, value):
        self._value = value
    def result(self):
        return self._value


class DummyExecutor:
    def __init__(self, *a, **k):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
    def submit(self, fn, *args, **kwargs):
        return DummyFuture(fn(*args, **kwargs))


def test_build_from_pages(monkeypatch):
    dummy_data = {'id': '1', 'content_embedding': [0.0], 'summary_embedding': [0.0],
                  'questions': ['q'], 'answers': ['a'], 'summary': 'text text text'}
    def fake_process_page(self, page, proc_executor=None):
        return DummyFuture(dummy_data)

    monkeypatch.setattr(sw.DatasetBuilder, 'process_page', fake_process_page)
    monkeypatch.setattr(sw, 'ThreadPoolExecutor', DummyExecutor)
    monkeypatch.setattr(sw, 'ProcessPoolExecutor', DummyExecutor)
    monkeypatch.setattr(sw, 'as_completed', lambda it: it)
    monkeypatch.setattr(sw, 'tqdm', lambda it, **k: it)

    builder = sw.DatasetBuilder()
    pages = [{'title': 't', 'lang': 'en'}]
    data = builder.build_from_pages(pages)
    assert data == [dummy_data]


def test_save_dataset_json_csv(tmp_path, monkeypatch):
    builder = sw.DatasetBuilder()
    monkeypatch.setattr(sw.Config, 'MIN_TEXT_LENGTH', 5)
    builder.dataset = [{
        'id': '1',
        'title': 't',
        'language': 'en',
        'category': 'c',
        'topic': 'ai',
        'subtopic': 'nlp',
        'keywords': [],
        'content': 'c'*20,
        'summary': 's'*20,
        'content_embedding': [0.1, 0.2],
        'summary_embedding': [0.1, 0.2],
        'questions': ['q'],
        'answers': ['a'],
        'created_at': 'now',
        'metadata': {}
    }]
    builder.save_dataset('json', output_dir=tmp_path)
    assert (tmp_path/'wikipedia_qa.json').exists()
    builder.save_dataset('csv', output_dir=tmp_path)
    assert (tmp_path/'wikipedia_qa.csv').exists()


def test_normalize_category_alias():
    assert sw.normalize_category('programacao') == 'Programação'


def test_main_uses_normalized_category(monkeypatch):
    called = []

    class DummyWiki:
        def __init__(self, lang):
            pass

        def get_category_members(self, category_name):
            called.append(category_name)
            return []

    class DummyBuilder:
        def build_from_pages(self, pages, *a, **k):
            return []

        def enhance_with_clustering(self):
            pass

        def save_dataset(self, format=None):
            pass

    monkeypatch.setattr(sw, 'WikipediaAdvanced', DummyWiki)
    monkeypatch.setattr(sw, 'DatasetBuilder', DummyBuilder)
    monkeypatch.setattr(sw.time, 'sleep', lambda *a, **k: None)

    sw.main(langs=['pt'], categories=['programacao'], fmt='json')

    assert called == ['Programação']

def test_search_category(monkeypatch):
    called = {}

    fake_cache = {}
    monkeypatch.setattr(sw, 'cache', SimpleNamespace(
        get=lambda k: fake_cache.get(k),
        set=lambda k, v: fake_cache.__setitem__(k, v),
        stats=lambda: {}
    ))

    class DummyResp:
        def __init__(self):
            pass
        def raise_for_status(self):
            pass
        status_code = 200
        def json(self):
            return {"query": {"search": [{"title": "Category:Computing"}]}}

    def fake_get(url, params=None, headers=None, timeout=None):
        called['params'] = params
        return DummyResp()

    monkeypatch.setattr(sw.requests, 'get', fake_get)
    result = sw.search_category('comp', 'en')
    assert result == 'Computing'
    assert called['params']['srnamespace'] == 14


def test_get_category_members_search(monkeypatch):
    wiki = sw.WikipediaAdvanced('en')
    fetch_calls = []

    dummy_member = SimpleNamespace(
        ns=sw.wikipediaapi.Namespace.MAIN,
        title='Page',
        fullurl='url'
    )
    dummy_cat = SimpleNamespace(
        exists=lambda: True,
        categorymembers={'p': dummy_member}
    )

    def fake_fetch(self, name):
        fetch_calls.append(name)
        if name == 'Missing':
            return None
        return dummy_cat

    monkeypatch.setattr(sw.WikipediaAdvanced, 'fetch_category', fake_fetch, raising=False)
    monkeypatch.setattr(sw, 'search_category', lambda keyword, lang: 'Found')

    members = wiki.get_category_members('Missing')
    assert members == [{
        'title': 'Page',
        'url': 'url',
        'lang': 'en',
        'category': 'Found',
        'depth': 0
    }]
    assert fetch_calls == ['Missing', 'Found']


def test_main_collects_pages_unaccented(monkeypatch):
    fake_cache = {}
    monkeypatch.setattr(sw, 'cache', SimpleNamespace(
        get=lambda k: fake_cache.get(k),
        set=lambda k, v: fake_cache.__setitem__(k, v),
        stats=lambda: {}
    ))
    original_norm = sw.normalize_category
    def fake_norm(name):
        if name == 'ciencia da computacao':
            return 'Ciência da computação'
        return original_norm(name)
    monkeypatch.setattr(sw, 'normalize_category', fake_norm)

    class DummyWiki:
        def __init__(self, lang):
            self.lang = lang

        def get_category_members(self, category_name):
            return [{
                'title': 'Page',
                'url': 'url',
                'lang': self.lang,
                'category': category_name,
                'depth': 0
            }]

    class DummyBuilder:
        def __init__(self):
            self.pages = []

        def build_from_pages(self, pages, *a, **k):
            self.pages.extend(pages)
            return []

        def enhance_with_clustering(self):
            pass

        def save_dataset(self, format=None):
            pass

    builder = DummyBuilder()

    monkeypatch.setattr(sw, 'WikipediaAdvanced', DummyWiki)
    monkeypatch.setattr(sw, 'DatasetBuilder', lambda: builder)
    monkeypatch.setattr(sw.time, 'sleep', lambda *a, **k: None)

    sw.main(langs=['pt'], categories=['ciencia da computacao'], fmt='json')

    assert len(builder.pages) == 1
    assert builder.pages[0]['category'] == 'Ciência da computação'


def test_main_collects_pages_alias(monkeypatch):
    fake_cache = {}
    monkeypatch.setattr(sw, 'cache', SimpleNamespace(
        get=lambda k: fake_cache.get(k),
        set=lambda k, v: fake_cache.__setitem__(k, v),
        stats=lambda: {}
    ))
    from unidecode import unidecode as real_unidecode
    monkeypatch.setattr(sw, 'unidecode', real_unidecode)
    class DummyWiki:
        def __init__(self, lang):
            self.lang = lang

        def get_category_members(self, category_name):
            return [{
                'title': 'AliasPage',
                'url': 'url',
                'lang': self.lang,
                'category': category_name,
                'depth': 0
            }]

    class DummyBuilder:
        def __init__(self):
            self.pages = []

        def build_from_pages(self, pages, *a, **k):
            self.pages.extend(pages)
            return []

        def enhance_with_clustering(self):
            pass

        def save_dataset(self, format=None):
            pass

    builder = DummyBuilder()

    monkeypatch.setattr(sw, 'WikipediaAdvanced', DummyWiki)
    monkeypatch.setattr(sw, 'DatasetBuilder', lambda: builder)
    monkeypatch.setattr(sw.time, 'sleep', lambda *a, **k: None)

    sw.main(langs=['pt'], categories=['programacao'], fmt='json')

    assert len(builder.pages) == 1
    assert builder.pages[0]['category'] == 'Programação'


def test_get_category_members_search_requests(monkeypatch):
    wiki = sw.WikipediaAdvanced('en')
    fetch_calls = []
    fake_cache = {}
    monkeypatch.setattr(sw, 'cache', SimpleNamespace(
        get=lambda k: fake_cache.get(k),
        set=lambda k, v: fake_cache.__setitem__(k, v),
        stats=lambda: {}
    ))

    dummy_member = SimpleNamespace(
        ns=sw.wikipediaapi.Namespace.MAIN,
        title='Page',
        fullurl='url'
    )
    dummy_cat = SimpleNamespace(
        exists=lambda: True,
        categorymembers={'p': dummy_member}
    )

    def fake_fetch(self, name):
        fetch_calls.append(name)
        if name == 'Found':
            return dummy_cat
        return None

    class DummyResp:
        def raise_for_status(self):
            pass
        status_code = 200

        def json(self):
            return {"query": {"search": [{"title": "Category:Found"}]}}

    def fake_get(url, params=None, headers=None, timeout=None):
        return DummyResp()

    monkeypatch.setattr(sw.WikipediaAdvanced, 'fetch_category', fake_fetch, raising=False)
    monkeypatch.setattr(sw.requests, 'get', fake_get)

    members = wiki.get_category_members('Missing')
    assert members == [{
        'title': 'Page',
        'url': 'url',
        'lang': 'en',
        'category': 'Found',
        'depth': 0
    }]
    assert fetch_calls == ['Missing', 'Found']


def test_rate_limiter_env(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_DELAY", "1.5")
    import importlib
    sw_reload = importlib.reload(sw)
    assert sw_reload.Config.RATE_LIMIT_DELAY == 1.5
    assert sw_reload.rate_limiter.delay == 1.5


def test_rate_limiter_backoff(monkeypatch):
    recorded = []
    rl = sw.RateLimiter(0.1)
    monkeypatch.setattr(sw.time, "sleep", lambda d: recorded.append(d))
    rl.wait()
    rl.record_error()
    rl.wait()
    assert recorded == [0.1, 0.2]
