import importlib
import json
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
wiki_mod = ModuleType('wikipediaapi')
wiki_mod.WikipediaException = Exception
wiki_mod.Namespace = SimpleNamespace(MAIN=0, CATEGORY=14)
wiki_mod.ExtractFormat = SimpleNamespace(HTML=0)
wiki_mod.WikipediaPage = object
wiki_mod.Wikipedia = lambda *a, **k: SimpleNamespace(page=lambda *a, **k: SimpleNamespace(exists=lambda: False), api=SimpleNamespace(article_url=lambda x: ""))
sys.modules.setdefault('wikipediaapi', wiki_mod)
aiohttp_stub = SimpleNamespace(
    ClientSession=object,
    ClientTimeout=lambda *a, **k: None,
    ClientError=Exception,
    ClientResponseError=Exception,
)
sys.modules.setdefault('aiohttp', aiohttp_stub)
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


def test_generate_qa_pairs_accepts_extra_metadata(monkeypatch):
    qb = sw.DatasetBuilder()
    monkeypatch.setattr(qb, '_generate_questions', lambda *a, **k: [])
    monkeypatch.setattr(qb, '_generate_answers', lambda *a, **k: [])
    monkeypatch.setattr(sw, 'extract_relations', lambda *a, **k: [])
    import numpy as np
    monkeypatch.setattr(qb.embedding_model, 'encode', lambda *a, **k: np.array([0.0]))
    result = qb.generate_qa_pairs(
        title='T',
        content='content',
        summary='sum',
        lang='en',
        category='c',
        extra_metadata={'wikidata_id': 'Q1', 'image_url': 'img'},
    )
    assert result['wikidata_id'] == 'Q1'
    assert result['image_url'] == 'img'

def test_advanced_clean_text_removes_html():
    raw = '<div>Example</div>\n[[1]]\n== Referências ==\nTexto'
    cleaned = sw.advanced_clean_text(raw, 'pt')
    assert '<div>' not in cleaned and 'Referências' not in cleaned

def test_advanced_clean_text_remove_stopwords(monkeypatch):
    class DummyTok:
        def __init__(self, text, stop):
            self.text = text
            self.is_stop = stop

    class DummyNLPStop:
        def __call__(self, text):
            return [
                DummyTok(tok, tok.lower() in {'the', 'and'})
                for tok in text.split()
            ]

    monkeypatch.setattr(sw.NLPProcessor, 'get_instance', classmethod(lambda cls, lang: DummyNLPStop()))
    cleaned = sw.advanced_clean_text('the quick and brown fox', 'en', remove_stopwords=True)
    assert cleaned == 'quick brown fox'

def test_extract_main_content():
    html = '<html><body><div id="mw-content-text">content</div><table></table></body></html>'
    result = sw.extract_main_content(html)
    assert 'content' in result and 'table' not in result


def test_extract_links_basic():
    html = (
        '<a href="/wiki/Page1">P1</a>'
        '<a href="https://example.com/wiki/Page2">P2</a>'
        '<a href="/other/Page3">P3</a>'
        '<a href="/wiki/Page4">P4</a>'
    )
    base = sw.get_base_url('en')
    links = sw.extract_links(html, base)
    assert links == [
        f"{sw.get_base_url('en')}/wiki/Page1",
        'https://example.com/wiki/Page2',
        f"{sw.get_base_url('en')}/wiki/Page4",
    ]


def test_get_links_from_category_page(monkeypatch):
    sample_html = '<a href="/wiki/Page">P</a>'
    called = {}

    def fake_fetch(title, lang):
        called['args'] = (title, lang)
        return sample_html

    monkeypatch.setattr(sw, 'fetch_html_content', fake_fetch)
    wiki = sw.WikipediaAdvanced('en')
    links = wiki.get_links_from_category_page('Test')

    assert called['args'] == ('Category:Test', 'en')
    assert links == [f"{sw.get_base_url('en')}/wiki/Page"]


def test_crawl_links_basic(monkeypatch):
    mapping = {
        'Start': ['P1', 'P2'],
        'P1': ['P3'],
        'P2': ['P4'],
    }

    def fake_fetch(title, lang):
        return title

    def fake_extract(html, base):
        return [f"{base}/wiki/{p}" for p in mapping.get(html, [])]

    monkeypatch.setattr(sw, 'fetch_html_content', fake_fetch)
    monkeypatch.setattr(sw, 'extract_links', fake_extract)

    wiki = sw.WikipediaAdvanced('en')
    result = wiki.crawl_links('Start', depth=1)
    base = sw.get_base_url('en')
    assert result == [
        {'title': 'P1', 'url': f'{base}/wiki/P1', 'lang': 'en', 'category': 'Start', 'depth': 0},
        {'title': 'P2', 'url': f'{base}/wiki/P2', 'lang': 'en', 'category': 'Start', 'depth': 0},
        {'title': 'P3', 'url': f'{base}/wiki/P3', 'lang': 'en', 'category': 'Start', 'depth': 1},
        {'title': 'P4', 'url': f'{base}/wiki/P4', 'lang': 'en', 'category': 'Start', 'depth': 1},
    ]


def test_crawl_links_async(monkeypatch):
    mapping = {
        'Start': ['P1'],
        'P1': ['P2'],
    }

    async def fake_fetch_async(title, lang):
        return title

    def fake_extract(html, base):
        return [f"{base}/wiki/{p}" for p in mapping.get(html, [])]

    monkeypatch.setattr(sw, 'fetch_html_content_async', fake_fetch_async)
    monkeypatch.setattr(sw, 'extract_links', fake_extract)

    wiki = sw.WikipediaAdvanced('en')
    import asyncio as aio
    result = aio.run(wiki.crawl_links_async('Start', depth=1))
    base = sw.get_base_url('en')
    assert result == [
        {'title': 'P1', 'url': f'{base}/wiki/P1', 'lang': 'en', 'category': 'Start', 'depth': 0},
        {'title': 'P2', 'url': f'{base}/wiki/P2', 'lang': 'en', 'category': 'Start', 'depth': 1},
    ]


def test_crawl_links_respects_limit(monkeypatch):
    mapping = {'A': ['B', 'C', 'D']}

    def fake_fetch(title, lang):
        return title

    def fake_extract(html, base):
        return [f"{base}/wiki/{p}" for p in mapping.get(html, [])]

    monkeypatch.setattr(sw, 'fetch_html_content', fake_fetch)
    monkeypatch.setattr(sw, 'extract_links', fake_extract)
    monkeypatch.setattr(sw.Config, 'MAX_PAGES_PER_CATEGORY', 2)

    wiki = sw.WikipediaAdvanced('en')
    result = wiki.crawl_links('A')
    assert len(result) == 2


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

    async def fake_fetch(url, headers=None, **kw):
        return 200, '<html></html>'

    async def fake_sleep(d):
        pass

    monkeypatch.setattr(sw, 'fetch_with_retry', fake_fetch)
    monkeypatch.setattr(sw.asyncio, 'sleep', fake_sleep)
    result = sw.fetch_html_content('Any', 'en')
    assert result == '<html></html>'


def test_fetch_html_content_error(monkeypatch):
    fake_cache = {}
    monkeypatch.setattr(sw, 'cache', SimpleNamespace(
        get=lambda k: fake_cache.get(k),
        set=lambda k, v, ttl=None: fake_cache.__setitem__(k, v),
        stats=lambda: {}
    ))

    async def fake_fetch(*a, **k):
        raise Exception('fail')

    async def fake_sleep(d):
        pass

    monkeypatch.setattr(sw, 'fetch_with_retry', fake_fetch)
    monkeypatch.setattr(sw.asyncio, 'sleep', fake_sleep)
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
    dummy_data = {
        'id': '1',
        'content_embedding': [0.0],
        'summary_embedding': [0.0],
        'questions': ['q'],
        'answers': ['a'],
        'summary': 'text text text',
        'wikidata_id': 'Q1',
        'image_url': 'img',
    }
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
    assert data[0]['wikidata_id'] == 'Q1'
    assert data[0]['image_url'] == 'img'


def test_build_from_pages_async(monkeypatch):
    dummy_data = {
        'id': '1',
        'content_embedding': [0.0],
        'summary_embedding': [0.0],
        'questions': ['q'],
        'answers': ['a'],
        'summary': 'text text text',
        'wikidata_id': 'Q1',
        'image_url': 'img'
    }

    async def fake_process_page_async(self, page, proc_executor=None):
        return DummyFuture(dummy_data)

    monkeypatch.setattr(sw.DatasetBuilder, 'process_page_async', fake_process_page_async)
    monkeypatch.setattr(sw, 'ProcessPoolExecutor', DummyExecutor)
    monkeypatch.setattr(sw, 'as_completed', lambda it: it)
    monkeypatch.setattr(sw, 'tqdm', lambda it, **k: it)

    builder = sw.DatasetBuilder()
    pages = [{'title': 't', 'lang': 'en'}]
    import asyncio as aio
    data = aio.run(builder.build_from_pages_async(pages))
    assert data == [dummy_data]
    assert data[0]['wikidata_id'] == 'Q1'
    assert data[0]['image_url'] == 'img'


def test_process_page_uses_clean_text(monkeypatch):
    class DummyPage:
        text = 'raw text'
        def exists(self):
            return True

    class DummyWiki:
        def __init__(self, lang):
            pass
        def fetch_page(self, title):
            return DummyPage()

    called = {}

    def fake_clean(text):
        called['clean'] = text
        return 'cleaned'

    def fake_adv(text, lang, remove_stopwords=False):
        called['adv'] = (text, lang, remove_stopwords)
        return 'x' * 151

    monkeypatch.setattr(sw, 'WikipediaAdvanced', DummyWiki)
    monkeypatch.setattr(sw, 'clean_text', fake_clean)
    monkeypatch.setattr(sw, 'advanced_clean_text', fake_adv)
    monkeypatch.setattr(sw.Config, 'REMOVE_STOPWORDS', True)
    monkeypatch.setattr(sw, 'summarize_text', lambda *a, **k: '')
    monkeypatch.setattr(sw.DatasetBuilder, 'generate_qa_pairs', lambda *a, **k: {})
    monkeypatch.setattr(sw, 'metrics', SimpleNamespace(
        scrape_success=SimpleNamespace(inc=lambda: None),
        scrape_error=SimpleNamespace(inc=lambda: None),
        pages_scraped_total=SimpleNamespace(inc=lambda: None),
        requests_failed_total=SimpleNamespace(inc=lambda: None),
        page_processing_seconds=SimpleNamespace(observe=lambda v: None),
    ))

    builder = sw.DatasetBuilder()
    res = builder.process_page({'title': 'T', 'lang': 'en'})

    assert res == {}
    assert called['clean'] == 'raw text'
    assert called['adv'] == ('cleaned', 'en', True)


def test_process_page_increments_counter(monkeypatch):
    class DummyPage:
        text = 't' * 200
        def exists(self):
            return True

    class DummyWiki:
        def __init__(self, lang):
            pass
        def fetch_page(self, title):
            return DummyPage()

    counts = {'pages': 0}

    def inc_pages():
        counts['pages'] += 1

    monkeypatch.setattr(sw.Config, 'MIN_TEXT_LENGTH', 10)
    monkeypatch.setattr(sw, 'WikipediaAdvanced', DummyWiki)
    monkeypatch.setattr(sw, 'clean_text', lambda t: t)
    monkeypatch.setattr(sw, 'advanced_clean_text', lambda t, lang, remove_stopwords=False: t)
    monkeypatch.setattr(sw, 'summarize_text', lambda *a, **k: '')
    monkeypatch.setattr(sw.DatasetBuilder, 'generate_qa_pairs', lambda *a, **k: {'ok': True})
    monkeypatch.setattr(sw, 'metrics', SimpleNamespace(
        scrape_success=SimpleNamespace(inc=lambda: None),
        scrape_error=SimpleNamespace(inc=lambda: None),
        pages_scraped_total=SimpleNamespace(inc=inc_pages),
        requests_failed_total=SimpleNamespace(inc=lambda: None),
        page_processing_seconds=SimpleNamespace(observe=lambda v: None),
    ))

    builder = sw.DatasetBuilder()
    res = builder.process_page({'title': 'T', 'lang': 'en'})

    assert res == {'ok': True}
    assert counts['pages'] == 1


def test_process_page_records_histogram(monkeypatch):
    class DummyPage:
        text = 't' * 200

        def exists(self):
            return True

    class DummyWiki:
        def __init__(self, lang):
            pass

        def fetch_page(self, title):
            return DummyPage()

    observed = {'count': 0}

    def observe(v):
        observed['count'] += 1

    monkeypatch.setattr(sw.Config, 'MIN_TEXT_LENGTH', 10)
    monkeypatch.setattr(sw, 'WikipediaAdvanced', DummyWiki)
    monkeypatch.setattr(sw, 'clean_text', lambda t: t)
    monkeypatch.setattr(sw, 'advanced_clean_text', lambda t, lang, remove_stopwords=False: t)
    monkeypatch.setattr(sw, 'summarize_text', lambda *a, **k: '')
    monkeypatch.setattr(sw.DatasetBuilder, 'generate_qa_pairs', lambda *a, **k: {'ok': True})
    monkeypatch.setattr(sw, 'metrics', SimpleNamespace(
        scrape_success=SimpleNamespace(inc=lambda: None),
        scrape_error=SimpleNamespace(inc=lambda: None),
        pages_scraped_total=SimpleNamespace(inc=lambda: None),
        requests_failed_total=SimpleNamespace(inc=lambda: None),
        page_processing_seconds=SimpleNamespace(observe=observe),
    ))

    builder = sw.DatasetBuilder()
    res = builder.process_page({'title': 'T', 'lang': 'en'})

    assert res == {'ok': True}
    assert observed['count'] == 1


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
        'relations': [],
        'created_at': 'now',
        'metadata': {}
    }]
    builder.save_dataset('json', output_dir=tmp_path)
    assert (tmp_path/'wikipedia_qa.json').exists()
    builder.save_dataset('csv', output_dir=tmp_path)
    assert (tmp_path/'wikipedia_qa.csv').exists()


def test_save_dataset_jsonl(tmp_path, monkeypatch):
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
        'relations': [],
        'created_at': 'now',
        'metadata': {}
    }]
    builder.save_dataset('jsonl', output_dir=tmp_path)
    jsonl_file = tmp_path / 'wikipedia_qa.jsonl'
    assert jsonl_file.exists()
    content = jsonl_file.read_text(encoding='utf-8').strip().splitlines()
    assert len(content) == 1
    record = json.loads(content[0])
    assert record['id'] == '1'


def test_save_dataset_qa_and_text(tmp_path, monkeypatch):
    builder = sw.DatasetBuilder()
    monkeypatch.setattr(sw.Config, 'MIN_TEXT_LENGTH', 5)
    builder.dataset = [{
        'id': '1',
        'title': 'T',
        'language': 'en',
        'category': 'c',
        'topic': 'ai',
        'subtopic': 'nlp',
        'keywords': [],
        'content': 'sample content '*2,
        'summary': 'summary text',
        'content_embedding': [0.1],
        'summary_embedding': [0.1],
        'questions': [{'text': 'q1'}],
        'answers': [{'text': 'a1'}],
        'relations': [],
        'created_at': 'now',
        'metadata': {}
    }]

    builder.save_dataset('qa', output_dir=tmp_path)
    qa_file = tmp_path / 'qa_pairs.json'
    assert qa_file.exists()
    data = json.loads(qa_file.read_text(encoding='utf-8'))
    assert data == [{'question': 'q1', 'answer': 'a1'}]

    builder.save_dataset('text', output_dir=tmp_path)
    txt_dir = tmp_path / 'text_corpus'
    assert (txt_dir / '1.txt').exists()


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
        set=lambda k, v, ttl=None: fake_cache.__setitem__(k, v),
        stats=lambda: {}
    ))

    async def fake_fetch(url, params=None, headers=None):
        called['params'] = params
        data = {"query": {"search": [{"title": "Category:Computing"}]}}
        return 200, json.dumps(data)

    async def fake_sleep(d):
        pass

    monkeypatch.setattr(sw, 'fetch_with_retry', fake_fetch)
    monkeypatch.setattr(sw.asyncio, 'sleep', fake_sleep)
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


def test_get_category_members_async(monkeypatch):
    wiki = sw.WikipediaAdvanced('en')
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
        return dummy_cat

    monkeypatch.setattr(sw.WikipediaAdvanced, 'fetch_category', fake_fetch, raising=False)
    import asyncio as aio
    members = aio.run(wiki.get_category_members_async('Any'))
    assert members == [{
        'title': 'Page',
        'url': 'url',
        'lang': 'en',
        'category': 'Any',
        'depth': 0
    }]


def test_main_collects_pages_unaccented(monkeypatch):
    fake_cache = {}
    monkeypatch.setattr(sw, 'cache', SimpleNamespace(
        get=lambda k: fake_cache.get(k),
        set=lambda k, v, ttl=None: fake_cache.__setitem__(k, v),
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
    monkeypatch.setattr(sw.metrics, 'start_metrics_server', lambda *a, **k: None)

    sw.main(langs=['pt'], categories=['ciencia da computacao'], fmt='json')

    assert len(builder.pages) == 1
    assert builder.pages[0]['category'] == 'Ciência da computação'


def test_main_collects_pages_alias(monkeypatch):
    fake_cache = {}
    monkeypatch.setattr(sw, 'cache', SimpleNamespace(
        get=lambda k: fake_cache.get(k),
        set=lambda k, v, ttl=None: fake_cache.__setitem__(k, v),
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
    monkeypatch.setattr(sw.metrics, 'start_metrics_server', lambda *a, **k: None)

    sw.main(langs=['pt'], categories=['programacao'], fmt='json')

    assert len(builder.pages) == 1
    assert builder.pages[0]['category'] == 'Programação'


def test_main_crawls_start_pages(monkeypatch):
    class DummyWiki:
        def __init__(self, lang):
            self.lang = lang

        def get_category_members(self, category_name):
            return []

        def crawl_links(self, start_page, depth):
            return [
                {
                    'title': 'P',
                    'url': 'url',
                    'lang': self.lang,
                    'category': start_page,
                    'depth': 0,
                }
            ]

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
    monkeypatch.setattr(sw.metrics, 'start_metrics_server', lambda *a, **k: None)

    sw.main(langs=['en'], start_pages=['Start'], depth=2)

    assert len(builder.pages) == 1
    assert builder.pages[0]['category'] == 'Start'


def test_main_async_crawls_start_pages(monkeypatch):
    class DummyWiki:
        def __init__(self, lang):
            self.lang = lang

        async def get_category_members_async(self, category_name):
            return []

        async def crawl_links_async(self, start_page, depth):
            return [
                {
                    'title': 'P',
                    'url': 'url',
                    'lang': self.lang,
                    'category': start_page,
                    'depth': 0,
                }
            ]

    class DummyBuilder:
        def __init__(self):
            self.pages = []

        def build_from_pages(self, pages, *a, **k):
            self.pages.extend(pages)
            return []

        async def build_from_pages_async(self, pages, *a, **k):
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
    monkeypatch.setattr(sw.metrics, 'start_metrics_server', lambda *a, **k: None)

    import asyncio as aio
    aio.run(sw.main_async(langs=['en'], start_pages=['Start'], depth=2))

    assert len(builder.pages) == 1
    assert builder.pages[0]['category'] == 'Start'


def test_get_category_members_search_requests(monkeypatch):
    wiki = sw.WikipediaAdvanced('en')
    fetch_calls = []
    fake_cache = {}
    monkeypatch.setattr(sw, 'cache', SimpleNamespace(
        get=lambda k: fake_cache.get(k),
        set=lambda k, v, ttl=None: fake_cache.__setitem__(k, v),
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

    def fake_fetch_category(self, name):
        fetch_calls.append(name)
        if name == 'Found':
            return dummy_cat
        return None

    async def fake_fetch_http(url, params=None, headers=None):
        data = {"query": {"search": [{"title": "Category:Found"}]}}
        return 200, json.dumps(data)

    monkeypatch.setattr(sw.WikipediaAdvanced, 'fetch_category', fake_fetch_category, raising=False)
    monkeypatch.setattr(sw, 'fetch_with_retry', fake_fetch_http)

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
    assert sw_reload.rate_limiter.min_delay == 1.5
    assert sw_reload.rate_limiter.max_delay == 1.5


def test_parallelism_env(monkeypatch):
    monkeypatch.setenv("MAX_THREADS", "7")
    monkeypatch.setenv("MAX_PROCESSES", "3")
    import importlib
    sw_reload = importlib.reload(sw)
    assert sw_reload.Config.MAX_THREADS == 7
    assert sw_reload.Config.MAX_PROCESSES == 3


def test_rate_limiter_backoff(monkeypatch):
    recorded = []
    rl = sw.RateLimiter(0.1)
    monkeypatch.setattr(sw.time, "sleep", lambda d: recorded.append(d))
    rl.wait()
    rl.record_error()
    rl.wait()
    assert recorded == [0.1, 0.2]


def test_rate_limiter_range_and_async(monkeypatch):
    recorded_sync = []
    recorded_async = []
    rl = sw.RateLimiter(0.1, 0.2)
    monkeypatch.setattr(sw.time, "sleep", lambda d: recorded_sync.append(d))
    async def fake_async_sleep(d):
        recorded_async.append(d)

    monkeypatch.setattr(sw.asyncio, "sleep", fake_async_sleep)
    monkeypatch.setattr(sw.random, "uniform", lambda a, b: b)
    rl.wait()

    async def run():
        await rl.async_wait()

    import asyncio as aio
    aio.run(run())

    assert recorded_sync == [0.2]
    assert recorded_async == [0.2]


def test_fetch_with_retry_failure_increments_counter(monkeypatch):
    import asyncio
    class DummyResp:
        async def __aenter__(self):
            raise sw.aiohttp.ClientError('fail')

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, *a, **k):
            return DummyResp()

    calls = {'fail': 0}

    class Marker(Exception):
        pass

    def inc_fail():
        calls['fail'] += 1
        raise Marker

    monkeypatch.setattr(sw.aiohttp, 'ClientSession', lambda *a, **k: DummySession())
    monkeypatch.setattr(sw.aiohttp, 'ClientTimeout', lambda *a, **k: None)
    monkeypatch.setattr(sw, 'log_failed_url', lambda url: None)
    monkeypatch.setattr(sw, 'metrics', SimpleNamespace(
        scrape_block=SimpleNamespace(inc=lambda: None),
        requests_failed_total=SimpleNamespace(inc=inc_fail),
        request_retries_total=SimpleNamespace(inc=lambda: None),
    ))

    import pytest

    with pytest.raises(Marker):
        asyncio.run(sw.fetch_with_retry('http://x'))

    assert calls['fail'] == 1


def test_fetch_with_retry_counts_retries(monkeypatch):
    import asyncio

    class DummyFail:
        async def __aenter__(self):
            raise sw.aiohttp.ClientError('fail')

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class DummySuccess:
        status = 200
        request_info = history = headers = None
        reason = 'OK'

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def text(self):
            return 'ok'

        def raise_for_status(self):
            pass

    attempts = [0]

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, *a, **k):
            attempts[0] += 1
            if attempts[0] == 1:
                return DummyFail()
            return DummySuccess()

    recorded = {'r': 0}

    def inc_retry():
        recorded['r'] += 1

    monkeypatch.setattr(sw.aiohttp, 'ClientSession', lambda *a, **k: DummySession())
    monkeypatch.setattr(sw.aiohttp, 'ClientTimeout', lambda *a, **k: None)
    monkeypatch.setattr(sw, 'log_failed_url', lambda url: None)
    monkeypatch.setattr(sw, 'metrics', SimpleNamespace(
        scrape_block=SimpleNamespace(inc=lambda: None),
        requests_failed_total=SimpleNamespace(inc=lambda: None),
        request_retries_total=SimpleNamespace(inc=inc_retry),
    ))

    status, text = asyncio.run(sw.fetch_with_retry('http://x', retries=2))

    assert status == 200
    assert recorded['r'] == 1


def test_rate_limiter_reset_after_success():
    rl = sw.RateLimiter(0.1)
    rl.record_error()
    assert rl.consecutive_failures == 1
    rl.record_success()
    assert rl.consecutive_failures == 0
    assert rl.min_delay == rl.base_min


def test_fetch_with_retry_429_increases_delay(monkeypatch):
    import asyncio
    rl = sw.RateLimiter(0.1)
    monkeypatch.setattr(sw, 'rate_limiter', rl)

    class DummyError(Exception):
        def __init__(self, status):
            super().__init__()
            self.status = status

    monkeypatch.setattr(sw.aiohttp, 'ClientResponseError', DummyError)

    class DummyResp:
        status = 429
        request_info = history = headers = None
        reason = 'Too Many'

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def text(self):
            return ''

        def raise_for_status(self):
            raise sw.aiohttp.ClientResponseError(self.status)

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, *a, **k):
            return DummyResp()

    monkeypatch.setattr(sw.aiohttp, 'ClientSession', lambda *a, **k: DummySession())
    monkeypatch.setattr(sw.aiohttp, 'ClientTimeout', lambda *a, **k: None)
    monkeypatch.setattr(sw, 'log_failed_url', lambda url: None)
    monkeypatch.setattr(sw, 'metrics', SimpleNamespace(
        scrape_block=SimpleNamespace(inc=lambda: None),
        requests_failed_total=SimpleNamespace(inc=lambda: None),
    ))

    import pytest

    with pytest.raises(sw.aiohttp.ClientResponseError):
        asyncio.run(sw.fetch_with_retry('http://x', retries=1))

    assert rl.consecutive_failures == 1
    assert rl.min_delay == 0.2


def test_fetch_with_retry_uses_proxy(monkeypatch):
    import asyncio

    class DummyResp:
        status = 200
        request_info = history = headers = None
        reason = 'OK'

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def text(self):
            return 'ok'

        def raise_for_status(self):
            pass

    class DummySession:
        def __init__(self):
            self.proxies = []
            self.calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, *a, **k):
            self.calls.append(k.get('proxy'))
            return DummyResp()

    session_inst = DummySession()

    monkeypatch.setattr(sw.aiohttp, 'ClientSession', lambda *a, **k: session_inst)
    monkeypatch.setattr(sw.aiohttp, 'ClientTimeout', lambda *a, **k: None)
    monkeypatch.setattr(sw.random, 'choice', lambda seq: seq[0])
    monkeypatch.setattr(sw.Config, 'PROXIES', ['http://p1'])
    monkeypatch.setattr(sw, 'log_failed_url', lambda url: None)
    monkeypatch.setattr(sw, 'metrics', SimpleNamespace(
        scrape_block=SimpleNamespace(inc=lambda: None),
        requests_failed_total=SimpleNamespace(inc=lambda: None),
        request_retries_total=SimpleNamespace(inc=lambda: None),
    ))

    asyncio.run(sw.fetch_with_retry('http://x'))

    assert session_inst.calls == ['http://p1']


def test_fetch_with_retry_semaphore_limits_concurrency(monkeypatch):
    import asyncio

    class DummyResp:
        status = 200
        request_info = history = headers = None
        reason = 'OK'

        async def __aenter__(self):
            await asyncio.sleep(0)
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def text(self):
            return 'ok'

        def raise_for_status(self):
            pass

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, *a, **k):
            return DummyResp()

    monkeypatch.setattr(sw.aiohttp, 'ClientSession', lambda *a, **k: DummySession())
    monkeypatch.setattr(sw.aiohttp, 'ClientTimeout', lambda *a, **k: None)
    monkeypatch.setattr(sw, 'log_failed_url', lambda url: None)
    monkeypatch.setattr(sw.Config, 'PROXIES', [])
    monkeypatch.setattr(sw, 'metrics', SimpleNamespace(
        scrape_block=SimpleNamespace(inc=lambda: None),
        requests_failed_total=SimpleNamespace(inc=lambda: None),
        request_retries_total=SimpleNamespace(inc=lambda: None),
    ))

    class DummySemaphore:
        def __init__(self, limit):
            self.limit = limit
            self.active = 0
            self.max_active = 0

        async def __aenter__(self):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active > self.limit:
                raise AssertionError('limit exceeded')

        async def __aexit__(self, exc_type, exc, tb):
            self.active -= 1

    sem = DummySemaphore(2)
    monkeypatch.setattr(sw, 'fetch_semaphore', sem)

    urls = ['http://a', 'http://b', 'http://c']
    asyncio.run(sw.fetch_all(urls))

    assert sem.max_active <= 2


def test_main_records_session_histogram(monkeypatch):
    observed = {'c': 0}

    def observe(v):
        observed['c'] += 1

    monkeypatch.setattr(sw.metrics, 'scrape_session_seconds', SimpleNamespace(observe=observe))
    monkeypatch.setattr(sw.metrics, 'start_metrics_server', lambda *a, **k: None)
    monkeypatch.setattr(sw.metrics, 'scrape_success', SimpleNamespace(inc=lambda: None))
    monkeypatch.setattr(sw.metrics, 'scrape_error', SimpleNamespace(inc=lambda: None))
    monkeypatch.setattr(sw.metrics, 'pages_scraped_total', SimpleNamespace(inc=lambda: None))
    monkeypatch.setattr(sw.metrics, 'requests_failed_total', SimpleNamespace(inc=lambda: None))
    monkeypatch.setattr(sw.metrics, 'request_retries_total', SimpleNamespace(inc=lambda: None))
    monkeypatch.setattr(sw.metrics, 'page_processing_seconds', SimpleNamespace(observe=lambda v: None))

    class DummyWiki:
        def __init__(self, lang):
            pass

        def get_category_members(self, category):
            return []

    class DummyBuilder:
        def build_from_pages(self, pages, progress_desc, client=None):
            pass

        def enhance_with_clustering(self):
            pass

        def save_dataset(self, format='all'):
            pass

    monkeypatch.setattr(sw, 'WikipediaAdvanced', DummyWiki)
    monkeypatch.setattr(sw, 'DatasetBuilder', DummyBuilder)
    sys.modules['plugins'] = SimpleNamespace(load_plugin=lambda name: SimpleNamespace(fetch_items=lambda l, c: [], parse_item=lambda i: None))

    sw.main(langs=['en'], categories=['prog'], fmt='json', rate_limit_delay=0, client=None)

    assert observed['c'] == 1


def test_crawl_links_uses_visited(monkeypatch):
    mapping = {
        'Start': ['A', 'B'],
        'A': ['C'],
        'B': [],
    }

    def fake_fetch(title, lang):
        return title

    def fake_extract(html, base):
        return [f"{base}/wiki/{p}" for p in mapping.get(html, [])]

    monkeypatch.setattr(sw, 'fetch_html_content', fake_fetch)
    monkeypatch.setattr(sw, 'extract_links', fake_extract)

    wiki = sw.WikipediaAdvanced('en')
    visited = {'A'}
    result = wiki.crawl_links('Start', depth=1, visited=visited)
    base = sw.get_base_url('en')
    assert all(r['title'] != 'A' for r in result)
    assert result == [
        {'title': 'B', 'url': f'{base}/wiki/B', 'lang': 'en', 'category': 'Start', 'depth': 0},
    ]


def test_crawl_links_async_uses_visited(monkeypatch):
    mapping = {
        'Start': ['A'],
        'A': ['B'],
    }

    async def fake_fetch_async(title, lang):
        return title

    def fake_extract(html, base):
        return [f"{base}/wiki/{p}" for p in mapping.get(html, [])]

    monkeypatch.setattr(sw, 'fetch_html_content_async', fake_fetch_async)
    monkeypatch.setattr(sw, 'extract_links', fake_extract)

    wiki = sw.WikipediaAdvanced('en')
    visited = {'A'}
    import asyncio as aio
    result = aio.run(wiki.crawl_links_async('Start', depth=1, visited=visited))
    base = sw.get_base_url('en')
    assert result == []


def test_advanced_clean_text_removes_templates_and_tables():
    raw = 'Intro {{Infobox}} text {| class="wikitable" | data |}'
    cleaned = sw.advanced_clean_text(raw, 'en')
    assert 'Infobox' not in cleaned and 'wikitable' not in cleaned
def test_advanced_clean_text_removes_see_also_section():
    raw = 'Something\n== See also ==\nOther info'
    cleaned = sw.advanced_clean_text(raw, 'en')
    assert 'See also' not in cleaned
