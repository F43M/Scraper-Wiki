from prometheus_client import Counter, Histogram, start_http_server

scrape_success = Counter(
    "scrape_success_total",
    "Total de páginas raspadas com sucesso"
)

scrape_error = Counter(
    "scrape_error_total",
    "Total de erros ao raspar páginas"
)

scrape_block = Counter(
    "scrape_block_total",
    "Total de bloqueios durante a raspagem"
)

# Total de registros de QA gerados com sucesso
pages_scraped_total = Counter(
    "pages_scraped_total",
    "Pages successfully scraped"
)

# Falhas ao executar requisições HTTP
requests_failed_total = Counter(
    "requests_failed_total",
    "HTTP request failures"
)

# Total de tentativas extras ao fazer requisições
request_retries_total = Counter(
    "request_retries_total",
    "HTTP request retries"
)

# Histogram of processing time per page
page_processing_seconds = Histogram(
    "page_processing_seconds",
    "Time spent processing a page in seconds"
)

# Duração completa de uma sessão de scraping
scrape_session_seconds = Histogram(
    "scrape_session_seconds",
    "Time spent in a full scraping session in seconds"
)


def start_metrics_server(port: int = 8001) -> None:
    """Inicia o servidor de métricas para Prometheus."""
    start_http_server(port)
