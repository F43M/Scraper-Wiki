from prometheus_client import Counter, start_http_server

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


def start_metrics_server(port: int = 8001) -> None:
    """Inicia o servidor de métricas para Prometheus."""
    start_http_server(port)
