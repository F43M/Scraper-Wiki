# 🚀 F43M Wikipedia Scraper Ultra Pro Max - GodMode++
# CEO: Fabio | Engenharia de Nível Industrial

import wikipediaapi

# Some versions of the wikipedia-api package do not expose
# ``WikipediaException``.  This attribute is referenced throughout the
# codebase when specifying retry logic for API calls.  To maintain
# compatibility with those versions we provide a simple fallback to the
# base ``Exception`` type when the attribute is missing.
if not hasattr(wikipediaapi, "WikipediaException"):
    class WikipediaException(Exception):
        """Fallback exception used when ``wikipediaapi`` lacks one."""

    wikipediaapi.WikipediaException = WikipediaException
import os
import re
import time
import json
import csv
import random
import logging
from tqdm import tqdm
from unidecode import unidecode
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datasets import Dataset, concatenate_datasets
from pathlib import Path
import hashlib
import pickle
import zlib
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import html2text
from typing import List, Dict, Tuple, Optional, Set, Protocol
from datetime import datetime
import multiprocessing
import signal
import backoff
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import sqlite3

# ============================
# 🔧 Configurações Avançadas
# ============================
class Config:
    # Idiomas suportados (prioridade ordenada)
    LANGUAGES = ['pt', 'en', 'es', 'fr', 'de', 'it', 'ja', 'zh']
    
    # Diretórios de saída
    OUTPUT_DIR = 'datasets_wikipedia_pro'
    CACHE_DIR = '.wiki_cache'
    LOG_DIR = 'logs'
    
    # Categorias avançadas com pesos
    CATEGORIES = {
        "Programação": 1.0,
        "Algoritmos": 0.9,
        "Linguagens de programação": 1.2,
        "Estruturas de dados": 0.95,
        "Engenharia de software": 1.1,
        "Ciência da computação": 0.85,
        "Desenvolvimento web": 1.15,
        "Banco de dados": 0.9,
        "Inteligência artificial": 1.3,
        "Segurança da informação": 0.95,
        "Machine Learning": 1.25,
        "Redes neurais": 1.1,
        "Visão computacional": 0.9,
        "Processamento de linguagem natural": 1.2,
        "Sistemas distribuídos": 0.85,
        "Computação quântica": 0.8,
        "Blockchain": 0.9,
        "Internet das Coisas": 0.85,
        "Realidade virtual": 0.8,
        "DevOps": 1.0
    }
    
    # Parâmetros avançados
    MAX_DEPTH = 3  # Profundidade máxima de navegação em categorias
    MAX_THREADS = multiprocessing.cpu_count() * 2
    MAX_PROCESSES = multiprocessing.cpu_count()
    RETRIES = 5
    TIMEOUT = 30
    RATE_LIMIT_DELAY = float(os.environ.get("RATE_LIMIT_DELAY", 0.5))
    MAX_PAGES_PER_CATEGORY = 1000
    MIN_TEXT_LENGTH = 150  # mínimo de caracteres para considerar uma página
    MAX_TEXT_LENGTH = 10000  # máximo de caracteres a extrair por página
    
    # Modelos de NLP
    NLP_MODELS = {
        'en': 'en_core_web_lg',
        'pt': 'pt_core_news_lg',
        'es': 'es_core_news_lg',
        'fr': 'fr_core_news_lg',
        'de': 'de_core_news_lg',
        'it': 'it_core_news_lg'
    }
    
    # Configuração de embeddings
    EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    # Configuração de sumarização
    SUMMARY_SENTENCES = 3
    
    # Configuração de clustering
    CLUSTER_K = 10
    
    # Proxies e headers
    PROXIES = []  # Lista de proxies rotativos pode ser adicionada
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]

    # Backend de cache: 'file' (padrão), 'redis' ou 'sqlite'
    CACHE_BACKEND = 'file'
    REDIS_URL = 'redis://localhost:6379/0'
    SQLITE_PATH = os.path.join(CACHE_DIR, 'cache.sqlite')
    CACHE_TTL: Optional[int] = int(os.environ.get("CACHE_TTL", "0")) or None
    
    @classmethod
    def get_random_user_agent(cls):
        return random.choice(cls.USER_AGENTS)


class RateLimiter:
    """Simple exponential backoff rate limiter."""

    def __init__(self, delay: float):
        self.base_delay = delay
        self.delay = delay

    def wait(self):
        time.sleep(self.delay)

    def record_error(self):
        self.delay *= 2

    def reset(self):
        self.delay = self.base_delay


# ============================
# 🗂 Categoria Normalização
# ============================

# Mapas de alias para nomes canônicos de categorias. As chaves devem estar
# normalizadas (sem acentos e em minúsculas) para facilitar a busca.
CATEGORY_ALIASES = {
    "programacao": "Programação",
}


def normalize_category(name: str) -> Optional[str]:
    """Retorna o nome canônico de uma categoria.

    A comparação ignora acentos e diferenças de maiúsculas/minúsculas.
    Se a categoria ou um de seus aliases for encontrado, devolve o nome
    oficial; caso contrário, ``None``.
    """

    normalized = unidecode(name).lower()

    for canonical in Config.CATEGORIES:
        if unidecode(canonical).lower() == normalized:
            return canonical

    return CATEGORY_ALIASES.get(normalized)

# ============================
# 📊 Logging Avançado
# ============================
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name, log_file, level=logging.INFO):
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    handler = logging.FileHandler(os.path.join(Config.LOG_DIR, log_file))
    handler.setFormatter(CustomFormatter())
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Adiciona também handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger('wiki_scraper', 'scraper.log')

# ============================
# 🧠 Cache Inteligente
# ============================

class CacheBackend(Protocol):
    def get(self, key: str):
        ...

    def set(self, key: str, data, ttl: Optional[int] = None):
        ...

    def stats(self) -> dict:
        ...


class FileCache(CacheBackend):
    def __init__(self):
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _get_cache_path(self, key: str) -> str:
        hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(Config.CACHE_DIR, f"{hash_key}.pkl.gz")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def get(self, key: str):
        cache_file = self._get_cache_path(key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    compressed_data = f.read()
                data = pickle.loads(zlib.decompress(compressed_data))
                self.hits += 1
                return data
            except Exception as e:
                logger.warning(f"Erro ao ler cache {key}: {e}")
                os.remove(cache_file)

        self.misses += 1
        return None

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def set(self, key: str, data, ttl: Optional[int] = None):
        cache_file = self._get_cache_path(key)
        try:
            compressed_data = zlib.compress(pickle.dumps(data))
            temp_file = cache_file + '.tmp'
            with open(temp_file, 'wb') as f:
                f.write(compressed_data)
            os.replace(temp_file, cache_file)
        except Exception as e:
            logger.error(f"Erro ao salvar cache {key}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def stats(self) -> dict:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


class RedisCache(CacheBackend):
    def __init__(self, url: str):
        try:
            import redis  # type: ignore
        except Exception as exc:  # pragma: no cover - import error
            raise ImportError("redis package required for RedisCache") from exc

        self.client = redis.Redis.from_url(url)
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        val = self.client.get(key)
        if val is not None:
            try:
                data = pickle.loads(zlib.decompress(val))
            except Exception:
                self.client.delete(key)
                self.misses += 1
                return None
            self.hits += 1
            return data
        self.misses += 1
        return None

    def set(self, key: str, data, ttl: Optional[int] = None):
        val = zlib.compress(pickle.dumps(data))
        if ttl:
            self.client.setex(key, ttl, val)
        else:
            self.client.set(key, val)

    def stats(self) -> dict:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


class SQLiteCache(CacheBackend):
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB, expires_at INTEGER)"
        )
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        cur = self.conn.execute("SELECT value, expires_at FROM cache WHERE key=?", (key,))
        row = cur.fetchone()
        if row:
            value, exp = row
            if exp is not None and exp < int(time.time()):
                self.conn.execute("DELETE FROM cache WHERE key=?", (key,))
                self.conn.commit()
                self.misses += 1
                return None
            try:
                data = pickle.loads(zlib.decompress(value))
            except Exception:
                self.conn.execute("DELETE FROM cache WHERE key=?", (key,))
                self.conn.commit()
                self.misses += 1
                return None
            self.hits += 1
            return data
        self.misses += 1
        return None

    def set(self, key: str, data, ttl: Optional[int] = None):
        val = zlib.compress(pickle.dumps(data))
        exp = int(time.time()) + ttl if ttl else None
        self.conn.execute(
            "REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)", (key, val, exp)
        )
        self.conn.commit()

    def stats(self) -> dict:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


def init_cache() -> CacheBackend:
    if Config.CACHE_BACKEND == 'redis':
        return RedisCache(Config.REDIS_URL)
    if Config.CACHE_BACKEND == 'sqlite':
        return SQLiteCache(Config.SQLITE_PATH)
    return FileCache()


cache: CacheBackend = init_cache()


def clear_cache() -> None:
    """Remove arquivos ou registros expirados do cache."""
    ttl = Config.CACHE_TTL
    if Config.CACHE_BACKEND == 'sqlite':
        if not os.path.exists(Config.SQLITE_PATH):
            return
        conn = sqlite3.connect(Config.SQLITE_PATH)
        conn.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (int(time.time()),),
        )
        conn.commit()
        conn.close()
    elif Config.CACHE_BACKEND == 'file':
        if ttl is None:
            return
        cutoff = time.time() - ttl
        path = Path(Config.CACHE_DIR)
        if path.exists():
            for p in path.glob("*.pkl.gz"):
                if p.stat().st_mtime < cutoff:
                    p.unlink()
    else:
        # Redis remove chaves expiradas automaticamente
        pass

# Global rate limiter for all network operations
rate_limiter = RateLimiter(Config.RATE_LIMIT_DELAY)

# ============================
# 🔍 Funções Avançadas de NLP
# ============================
class NLPProcessor:
    _instances = {}
    
    @classmethod
    def get_instance(cls, lang: str):
        if lang not in cls._instances:
            if lang in Config.NLP_MODELS:
                try:
                    cls._instances[lang] = spacy.load(Config.NLP_MODELS[lang])
                    logger.info(f"Carregado modelo NLP para {lang}")
                except OSError:
                    logger.warning(
                        f"Modelo NLP para {lang} não encontrado, tentando modelo pequeno"
                    )
                    try:
                        cls._instances[lang] = spacy.load(
                            Config.NLP_MODELS[lang].replace("_lg", "_sm")
                        )
                    except OSError:
                        logger.warning(
                            f"Modelos lg e sm para {lang} indisponíveis, usando 'en_core_web_sm'"
                        )
                        cls._instances[lang] = spacy.load("en_core_web_sm")
            else:
                logger.warning(
                    f"Modelo NLP para {lang} não configurado, usando inglês"
                )
                cls._instances[lang] = spacy.load(Config.NLP_MODELS["en"])
        return cls._instances[lang]
    
    @classmethod
    def get_embedding_model(cls):
        if not hasattr(cls, '_embedding_model'):
            cls._embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        return cls._embedding_model

def extract_keywords(text: str, lang: str = 'en', n: int = 10) -> List[str]:
    try:
        nlp = NLPProcessor.get_instance(lang)
        doc = nlp(text)
        
        # Filtra substantivos e nomes próprios
        keywords = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
        
        # Remove duplicatas e conta frequência
        freq = {}
        for word in keywords:
            word_lower = word.lower()
            if word_lower in freq:
                freq[word_lower] += 1
            else:
                freq[word_lower] = 1
        
        # Ordena por frequência e pega os top N
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in sorted_keywords[:n]]
    
    except Exception as e:
        logger.error(f"Erro ao extrair keywords: {e}")
        return []

def summarize_text(text: str, lang: str = 'en') -> str:
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(lang))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, Config.SUMMARY_SENTENCES)
        return ' '.join([str(sentence) for sentence in summary])
    except Exception as e:
        logger.error(f"Erro ao sumarizar texto: {e}")
        return text[:Config.MIN_TEXT_LENGTH] if len(text) > Config.MIN_TEXT_LENGTH else text

def cluster_texts(texts: List[str], k: int = Config.CLUSTER_K) -> np.ndarray:
    try:
        model = NLPProcessor.get_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Normaliza os embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        
        # Clusteriza usando K-means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(normalized_embeddings)
        
        return clusters
    except Exception as e:
        logger.error(f"Erro no clustering: {e}")
        return np.zeros(len(texts))

# ============================
# 🧹 Limpeza Avançada de Texto
# ============================
def advanced_clean_text(text: str, lang: str = 'en') -> str:
    try:
        # Remove HTML
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove caracteres especiais, mantendo acentos quando relevante
        if lang in ['en', 'de']:
            text = unidecode(text)
        
        # Remove padrões específicos da Wikipedia
        text = re.sub(r'\[\d+\]', '', text)  # Referências [1], [2], etc.
        text = re.sub(r'\b(ver também|see also|véase también|voir aussi)\b.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove templates wiki
        text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)  # Remove tables
        
        # Remove seções específicas
        sections_to_remove = [
            'referências', 'references', 'referencias',
            'bibliografia', 'bibliography', 'bibliografía',
            'ligações externas', 'external links', 'enlaces externos'
        ]
        for section in sections_to_remove:
            text = re.sub(fr'==\s*{section}\s*==.*', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text.strip()
    except Exception as e:
        logger.error(f"Erro na limpeza de texto: {e}")
        return text

def extract_main_content(page_html: str) -> str:
    try:
        soup = BeautifulSoup(page_html, 'html.parser')
        
        # Remove elementos indesejados
        for element in soup.find_all(['table', 'div.infobox', 'span.reference', 'ol.references', 
                                    'div.navbox', 'div.hatnote', 'div.thumb', 'div.notice']):
            element.decompose()
        
        # Extrai conteúdo principal
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if content_div:
            return str(content_div)
        return page_html
    except Exception as e:
        logger.error(f"Erro ao extrair conteúdo principal: {e}")
        return page_html

def fetch_html_content(title: str, lang: str) -> str:
    """Retrieve the raw HTML for a Wikipedia page using the REST API."""
    cache_key = f"html_{lang}_{title}"
    cached = cache.get(cache_key)

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, wikipediaapi.WikipediaException),
        max_tries=Config.RETRIES,
    )
    def _request() -> str:
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}"
        response = requests.get(
            url,
            headers={"User-Agent": Config.get_random_user_agent()},
            timeout=Config.TIMEOUT,
        )
        response.raise_for_status()
        return response.text

    try:
        html = _request()
        cache.set(cache_key, html, ttl=Config.CACHE_TTL)
        return html
    except Exception as e:
        logger.error(f"Erro ao buscar HTML para {title}: {e}")
        if cached is not None:
            return cached
        return ""

def search_category(keyword: str, lang: str) -> Optional[str]:
    """Search for a similar category name using the Wikipedia API."""
    cache_key = f"search_category_{lang}_{keyword}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srnamespace": 14,
        "srsearch": keyword,
        "format": "json",
    }

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, wikipediaapi.WikipediaException),
        max_tries=Config.RETRIES,
    )
    def _request():
        rate_limiter.wait()
        resp = requests.get(
            url,
            params=params,
            headers={"User-Agent": Config.get_random_user_agent()},
            timeout=Config.TIMEOUT,
        )
        if resp.status_code == 429:
            rate_limiter.record_error()
        resp.raise_for_status()
        return resp.json()

    try:
        data = _request()
        results = data.get("query", {}).get("search", [])
        if results:
            title = results[0].get("title", "")
            title = title.replace("Category:", "").replace("Categoria:", "")
            cache.set(cache_key, title, ttl=Config.CACHE_TTL)
            return title
    except Exception as e:  # pragma: no cover - network issues
        rate_limiter.record_error()
        logger.error(f"Erro ao buscar categorias para {keyword}: {e}")

    fallback = cache.get(cache_key)
    return fallback

# ============================
# 🔗 Coletor Avançado com Retry
# ============================
class WikipediaAdvanced:
    def __init__(self, lang: str):
        self.lang = lang
        self.wiki = wikipediaapi.Wikipedia(
            language=lang,
            extract_format=wikipediaapi.ExtractFormat.HTML,
            user_agent=Config.get_random_user_agent()
        )
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': Config.get_random_user_agent()})

    def _prepare_session(self):
        """Refresh User-Agent and proxy settings before a request."""
        self.session.headers['User-Agent'] = Config.get_random_user_agent()
        if Config.PROXIES:
            self.session.proxies = random.choice(Config.PROXIES)
        else:
            self.session.proxies = {}
    
    @backoff.on_exception(backoff.expo, 
                         (requests.exceptions.RequestException, 
                          wikipediaapi.WikipediaException),
                         max_tries=Config.RETRIES)
    def fetch_page(self, page_title: str) -> Optional[wikipediaapi.WikipediaPage]:
        cache_key = f"page_{self.lang}_{page_title}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        self._prepare_session()
        rate_limiter.wait()

        try:
            page = self.wiki.page(page_title)
            if page.exists():
                # Melhora a qualidade do conteúdo
                page._fullurl = self.wiki.api.article_url(page_title)
                rate_limiter.wait()
                page._html = fetch_html_content(page_title, self.lang)
                page._html = extract_main_content(page._html)

                cache.set(cache_key, page, ttl=Config.CACHE_TTL)
                return page
        except Exception as e:
            rate_limiter.record_error()
            logger.error(f"Erro ao buscar página {page_title}: {e}")
            raise
        
        return None
    
    @backoff.on_exception(backoff.expo, 
                         (requests.exceptions.RequestException, 
                          wikipediaapi.WikipediaException),
                         max_tries=Config.RETRIES)
    def fetch_category(self, category_name: str) -> Optional[wikipediaapi.WikipediaPage]:
        category_title = f"Category:{category_name}" if self.lang != 'pt' else f"Categoria:{category_name}"
        self._prepare_session()
        rate_limiter.wait()
        return self.fetch_page(category_title)
    
    def get_related_pages(self, page_title: str, limit: int = 10) -> List[dict]:
        cache_key = f"related_{self.lang}_{page_title}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        self._prepare_session()

        @backoff.on_exception(
            backoff.expo,
            (requests.exceptions.RequestException, wikipediaapi.WikipediaException),
            max_tries=Config.RETRIES,
        )
        def _request():
            url = f"https://{self.lang}.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'titles': page_title,
                'prop': 'links',
                'pllimit': limit,
                'format': 'json'
            }
            response = self.session.get(url, params=params, timeout=Config.TIMEOUT)
            response.raise_for_status()
            return response.json()

        try:
            data = _request()

            pages = data.get('query', {}).get('pages', {})
            links = []

            for page in pages.values():
                for link in page.get('links', []):
                    if 'ns' in link and link['ns'] == 0:  # Only main namespace
                        links.append({
                            'title': link['title'],
                            'lang': self.lang
                        })

            cache.set(cache_key, links, ttl=Config.CACHE_TTL)
            return links
        except Exception as e:
            logger.error(f"Erro ao buscar páginas relacionadas para {page_title}: {e}")
            fallback = cache.get(cache_key)
            if fallback is not None:
                return fallback
            return []
    
    def get_category_members(self, category_name: str, depth: int = 0, visited: Optional[Set[str]] = None) -> List[dict]:
        if visited is None:
            visited = set()
        
        category = self.fetch_category(category_name)
        if not category or not category.exists():
            alt = search_category(category_name, self.lang)
            if alt:
                category_name = alt
                category = self.fetch_category(category_name)

        if not category or not category.exists():
            logger.warning(f"Categoria não encontrada: {category_name}")
            return []
        
        members = []
        for member in category.categorymembers.values():
            if member.title not in visited:
                visited.add(member.title)
                
                if member.ns == wikipediaapi.Namespace.CATEGORY and depth < Config.MAX_DEPTH:
                    sub_members = self.get_category_members(
                        member.title.replace("Category:", "").replace("Categoria:", ""),
                        depth + 1,
                        visited
                    )
                    members.extend(sub_members)
                elif member.ns == wikipediaapi.Namespace.MAIN:
                    members.append({
                        'title': member.title,
                        'url': member.fullurl,
                        'lang': self.lang,
                        'category': category_name,
                        'depth': depth
                    })
                
                if len(members) >= Config.MAX_PAGES_PER_CATEGORY:
                    break
        
        return members

# ============================
# 🏗️ Builder de Dataset Profissional
# ============================

def cpu_process_page(title: str, content: str, lang: str, category: str) -> dict:
    """Executes CPU intensive operations for a page."""
    builder = DatasetBuilder()
    summary = summarize_text(content, lang)
    return builder.generate_qa_pairs(
        title=title,
        content=content,
        summary=summary,
        lang=lang,
        category=category,
    )

class DatasetBuilder:
    def __init__(self):
        self.embedding_model = NLPProcessor.get_embedding_model()
        self.dataset = []
        self.qa_pairs = []

    def _update_progress(self):
        """Update progress information in logs/progress.json"""
        try:
            os.makedirs(Config.LOG_DIR, exist_ok=True)
            progress_file = os.path.join(Config.LOG_DIR, 'progress.json')
            temp_file = progress_file + '.tmp'

            clusters = sorted({item.get('cluster') for item in self.dataset if 'cluster' in item})
            topics = sorted({item.get('topic') for item in self.dataset if item.get('topic')})
            languages = sorted({item.get('language') for item in self.dataset if item.get('language')})

            progress = {
                'pages_processed': len(self.dataset),
                'clusters': clusters,
                'topics': topics,
                'languages': languages,
            }

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, progress_file)
        except Exception as e:
            logger.error(f"Erro ao atualizar progresso: {e}")

    def process_page(
        self,
        page_info: dict,
        proc_executor: Optional[ProcessPoolExecutor] = None
    ) -> Optional[object]:
        try:
            wiki = WikipediaAdvanced(page_info['lang'])
            page = wiki.fetch_page(page_info['title'])

            if not page or not page.exists():
                return None

            # Extrai e limpa o texto
            clean_content = advanced_clean_text(page.text, page_info['lang'])

            if len(clean_content) < Config.MIN_TEXT_LENGTH:
                return None

            if proc_executor:
                return proc_executor.submit(
                    cpu_process_page,
                    page_info['title'],
                    clean_content,
                    page_info['lang'],
                    page_info.get('category', '')
                )

            # Sumariza o conteúdo
            summary = summarize_text(clean_content, page_info['lang'])

            # Gera QA pairs avançadas
            qa_data = self.generate_qa_pairs(
                title=page_info['title'],
                content=clean_content,
                summary=summary,
                lang=page_info['lang'],
                category=page_info.get('category', '')
            )

            return qa_data
        except Exception as e:
            logger.error(
                f"Erro ao processar página {page_info.get('title', '')}: {e}"
            )
            return None
    
    def generate_qa_pairs(self, title: str, content: str, summary: str, lang: str, category: str) -> dict:
        # Extrai keywords para gerar perguntas variadas
        keywords = extract_keywords(content, lang)
        
        # Gera múltiplas perguntas baseadas no conteúdo
        questions = self._generate_questions(title, content, lang, keywords)
        
        # Gera respostas completas
        answers = self._generate_answers(content, summary, lang)
        
        # Cria embeddings para busca semântica
        content_embedding = self.embedding_model.encode(content, show_progress_bar=False)
        summary_embedding = self.embedding_model.encode(summary, show_progress_bar=False)
        
        # Classificação avançada de tópicos
        topic, subtopic = self._classify_topic(title, content, lang)
        
        return {
            'id': hashlib.md5(f"{title}_{lang}".encode('utf-8')).hexdigest(),
            'title': title,
            'language': lang,
            'category': category,
            'topic': topic,
            'subtopic': subtopic,
            'keywords': keywords,
            'content': content,
            'summary': summary,
            'content_embedding': content_embedding.tolist(),
            'summary_embedding': summary_embedding.tolist(),
            'questions': questions,
            'answers': answers,
            'created_at': datetime.utcnow().isoformat(),
            'metadata': {
                'length': len(content),
                'source': 'wikipedia',
                'source_url': f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            }
        }
    
    def _generate_questions(self, title: str, content: str, lang: str, keywords: List[str]) -> List[dict]:
        questions = []
        
        # Pergunta básica sobre o título
        base_question = {
            'text': self._translate_question(f"What is {title}?", lang),
            'type': 'definition',
            'difficulty': 'easy'
        }
        questions.append(base_question)
        
        # Perguntas baseadas em keywords
        for keyword in keywords[:5]:  # Limita a 5 perguntas por keyword
            question_types = [
                (f"How does {keyword} relate to {title}?", 'relation', 'medium'),
                (f"What is the role of {keyword} in {title}?", 'role', 'medium'),
                (f"Can you explain {keyword} in the context of {title}?", 'explanation', 'hard')
            ]
            
            for q_text, q_type, q_diff in question_types:
                questions.append({
                    'text': self._translate_question(q_text, lang),
                    'type': q_type,
                    'difficulty': q_diff
                })
        
        # Perguntas baseadas em conteúdo (usando NLP)
        try:
            nlp = NLPProcessor.get_instance(lang)
            doc = nlp(content)
            
            # Perguntas sobre entidades nomeadas
            for ent in doc.ents[:5]:  # Limita a 5 entidades
                if ent.label_ in ['PERSON', 'ORG', 'LOC', 'PRODUCT']:
                    questions.append({
                        'text': self._translate_question(f"What is the significance of {ent.text} in {title}?", lang),
                        'type': 'significance',
                        'difficulty': 'medium'
                    })
            
            # Perguntas sobre verbos/ações
            for sent in doc.sents[:3]:  # Analisa as primeiras 3 sentenças
                root = [token for token in sent if token.dep_ == 'ROOT']
                if root:
                    verb = root[0].lemma_
                    questions.append({
                        'text': self._translate_question(f"How does {title} {verb}?", lang),
                        'type': 'process',
                        'difficulty': 'hard'
                    })
        except Exception as e:
            logger.error(f"Erro ao gerar perguntas NLP para {title}: {e}")
        
        return questions
    
    def _generate_answers(self, content: str, summary: str, lang: str) -> List[dict]:
        answers = []
        
        # Resposta resumida
        answers.append({
            'text': summary,
            'type': 'summary',
            'length': 'short'
        })
        
        # Resposta completa
        answers.append({
            'text': content[:Config.MAX_TEXT_LENGTH],  # Limita o tamanho
            'type': 'detailed',
            'length': 'long'
        })
        
        # Respostas específicas por parágrafo
        paragraphs = [p for p in content.split('\n') if len(p.strip()) > 0]
        for para in paragraphs[:3]:  # Limita a 3 parágrafos
            answers.append({
                'text': para,
                'type': 'paragraph',
                'length': 'medium'
            })
        
        return answers
    
    def _translate_question(self, question: str, lang: str) -> str:
        translations = {
            'pt': {
                'What is': 'O que é',
                'How does': 'Como',
                'relate to': 'se relaciona com',
                'What is the role of': 'Qual é o papel de',
                'Can you explain': 'Você pode explicar',
                'in the context of': 'no contexto de',
                'What is the significance of': 'Qual é a importância de'
            },
            'es': {
                'What is': 'Qué es',
                'How does': 'Cómo',
                'relate to': 'se relaciona con',
                'What is the role of': 'Cuál es el papel de',
                'Can you explain': 'Puedes explicar',
                'in the context of': 'en el contexto de',
                'What is the significance of': 'Cuál es la importancia de'
            },
            'fr': {
                'What is': 'Qu\'est-ce que',
                'How does': 'Comment',
                'relate to': 'se rapporte à',
                'What is the role of': 'Quel est le rôle de',
                'Can you explain': 'Pouvez-vous expliquer',
                'in the context of': 'dans le contexte de',
                'What is the significance of': 'Quelle est l\'importance de'
            }
        }
        
        if lang in translations:
            for eng, trans in translations[lang].items():
                question = question.replace(eng, trans)
        
        return question
    
    def _classify_topic(self, title: str, content: str, lang: str) -> Tuple[str, str]:
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Mapeamento de tópicos principais e subtópicos
        topics = {
            'ai': {
                'keywords': ['inteligência artificial', 'machine learning', 'ai', 'deep learning', 'redes neurais'],
                'subtopics': {
                    'nlp': ['processamento de linguagem natural', 'pln', 'nlu', 'nlp'],
                    'vision': ['visão computacional', 'computer vision', 'reconhecimento de imagem'],
                    'robotics': ['robótica', 'robots', 'autonomous systems']
                }
            },
            'web': {
                'keywords': ['desenvolvimento web', 'frontend', 'backend', 'full stack', 'javascript', 'html', 'css'],
                'subtopics': {
                    'frontend': ['frontend', 'interface', 'ui', 'ux', 'react', 'vue', 'angular'],
                    'backend': ['backend', 'servidor', 'api', 'rest', 'graphql', 'node.js'],
                    'fullstack': ['full stack', 'full-stack', 'mern', 'mean']
                }
            },
            'data': {
                'keywords': ['banco de dados', 'big data', 'data science', 'data analytics', 'sql', 'nosql'],
                'subtopics': {
                    'sql': ['sql', 'relacional', 'mysql', 'postgresql', 'oracle'],
                    'nosql': ['nosql', 'mongodb', 'cassandra', 'redis', 'elasticsearch'],
                    'bigdata': ['big data', 'hadoop', 'spark', 'data lake']
                }
            }
        }
        
        # Tenta encontrar o tópico principal
        main_topic = 'engenharia de software'
        subtopic = 'geral'
        
        for topic, data in topics.items():
            if any(kw in title_lower or kw in content_lower for kw in data['keywords']):
                main_topic = topic
                
                # Tenta encontrar subtópico
                for sub, sub_kws in data['subtopics'].items():
                    if any(skw in title_lower or skw in content_lower for skw in sub_kws):
                        subtopic = sub
                        break
                break
        
        return (main_topic, subtopic)
    
    def build_from_pages(self, pages: List[dict], progress_desc: str = "Processando páginas") -> List[dict]:
        cpu_futures = []
        with ThreadPoolExecutor(max_workers=Config.MAX_THREADS) as th_executor, \
                ProcessPoolExecutor(max_workers=Config.MAX_PROCESSES) as pr_executor:

            page_futures = {
                th_executor.submit(self.process_page, page, pr_executor): page
                for page in pages
            }

            for future in tqdm(as_completed(page_futures), total=len(page_futures), desc=progress_desc):
                cpu_future = future.result()
                if cpu_future:
                    cpu_futures.append(cpu_future)

            for future in tqdm(as_completed(cpu_futures), total=len(cpu_futures), desc="Processando conteúdo"):
                result = future.result()
                if result:
                    self.dataset.append(result)
                    self._update_progress()

        return self.dataset
    
    def enhance_with_clustering(self):
        if not self.dataset:
            return
        
        # Clusteriza baseado nos embeddings de conteúdo
        texts = [item['content'] for item in self.dataset]
        clusters = cluster_texts(texts)
        
        # Adiciona clusters ao dataset
        for i, item in enumerate(self.dataset):
            item['cluster'] = int(clusters[i])
    
    def save_dataset(self, format: str = 'all', output_dir: str = Config.OUTPUT_DIR):
        os.makedirs(output_dir, exist_ok=True)

        if not self.dataset:
            logger.warning("Nenhum dado para salvar")
            return

        # Validação dos registros antes de salvar
        validated_data = []
        for item in self.dataset:
            valid = True

            # Checa se embeddings contêm apenas números finitos
            if not np.all(np.isfinite(item.get('content_embedding', []))):
                logger.warning(
                    f"content_embedding inválido para {item.get('id', 'desconhecido')}"
                )
                valid = False

            if not np.all(np.isfinite(item.get('summary_embedding', []))):
                logger.warning(
                    f"summary_embedding inválido para {item.get('id', 'desconhecido')}"
                )
                valid = False

            # Checa presença de perguntas e respostas
            if not item.get('questions'):
                logger.warning(
                    f"Registro {item.get('id', 'desconhecido')} sem perguntas"
                )
                valid = False

            if not item.get('answers'):
                logger.warning(
                    f"Registro {item.get('id', 'desconhecido')} sem respostas"
                )
                valid = False

            # Verifica tamanho do resumo
            summary_text = item.get('summary', '')
            if len(summary_text) < Config.MIN_TEXT_LENGTH:
                logger.warning(
                    f"Resumo muito curto para {item.get('id', 'desconhecido')}"
                )
                valid = False

            if valid:
                validated_data.append(item)

        if not validated_data:
            logger.warning("Nenhum registro válido para salvar")
            return

        # Ordena por idioma e tópico
        sorted_data = sorted(validated_data, key=lambda x: (x['language'], x['topic']))
        
        # Salva em múltiplos formatos
        if format in ['all', 'json']:
            json_file = os.path.join(output_dir, 'wikipedia_qa.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Dataset salvo em JSON: {json_file}")
        
        if format in ['all', 'csv']:
            csv_file = os.path.join(output_dir, 'wikipedia_qa.csv')
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted_data[0].keys())
                writer.writeheader()
                rows = []
                for row in sorted_data:
                    converted = {
                        k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()
                    }
                    rows.append(converted)
                writer.writerows(rows)
            logger.info(f"Dataset salvo em CSV: {csv_file}")
        
        if format in ['all', 'parquet']:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                parquet_file = os.path.join(output_dir, 'wikipedia_qa.parquet')
                table = pa.Table.from_pylist(sorted_data)
                pq.write_table(table, parquet_file)
                logger.info(f"Dataset salvo em Parquet: {parquet_file}")
            except ImportError:
                logger.warning("PyArrow não instalado, pulando salvamento Parquet")
        
        if format in ['all', 'hf']:
            try:
                hf_dataset = Dataset.from_list(sorted_data)
                hf_dataset.save_to_disk(os.path.join(output_dir, 'huggingface'))
                logger.info(f"Dataset salvo para HuggingFace: {os.path.join(output_dir, 'huggingface')}")
            except Exception as e:
                logger.error(f"Erro ao salvar dataset HuggingFace: {e}")

# ============================
# 🚦 Pipeline de Execução Principal
# ============================
def main(langs: Optional[List[str]] = None,
         categories: Optional[List[str]] = None,
         fmt: str = "all",
         rate_limit_delay: Optional[float] = None):
    """Gera o dataset utilizando os parâmetros fornecidos."""
    logger.info("🚀 Iniciando Wikipedia Scraper Ultra Pro Max - GodMode++")

    if rate_limit_delay is not None:
        Config.RATE_LIMIT_DELAY = rate_limit_delay
        rate_limiter.base_delay = rate_limit_delay
        rate_limiter.reset()

    languages = langs or Config.LANGUAGES
    cats = Config.CATEGORIES
    if categories:
        normalized = [normalize_category(c) or c for c in categories]
        cats = {c: Config.CATEGORIES.get(c, 1.0) for c in normalized}

    builder = DatasetBuilder()

    all_pages: List[dict] = []
    for lang in languages:
        logger.info(f"🌐 Processando idioma: {lang.upper()}")
        wiki = WikipediaAdvanced(lang)

        for category, weight in cats.items():
            logger.info(f"🔍 Buscando na categoria: {category} (peso: {weight})")

            pages = wiki.get_category_members(category)
            logger.info(f"📄 Páginas encontradas em {category}: {len(pages)}")

            for page in pages:
                page['weight'] = weight

            all_pages.extend(pages)
            time.sleep(Config.RATE_LIMIT_DELAY * 2)

    logger.info(f"📚 Total de páginas coletadas: {len(all_pages)}")

    builder.build_from_pages(all_pages, "Construindo dataset")

    logger.info("🧠 Aplicando técnicas avançadas de NLP...")
    builder.enhance_with_clustering()

    logger.info("💾 Salvando dataset completo...")
    builder.save_dataset(format=fmt)

    logger.info("✅ Dataset finalizado com sucesso!")
    logger.info(f"📊 Estatísticas de cache: {cache.stats()}")

if __name__ == "__main__":
    main()
