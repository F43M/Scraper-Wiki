# Scraper-Wiki
Scraper para criacao de datasets para fine tuning e treinamento de modelos de inteligencia artificial

## Instalação

Use o arquivo `requirements.txt` para instalar as dependências:

O projeto requer **Python 3.11** ou superior.

```bash
pip install -r requirements.txt
```

## Uso via linha de comando

Utilize o script `cli.py` para interagir com o scraper. Para executar uma coleta imediatamente:

```bash
python cli.py scrape --lang pt --category "Programação" --format json
```

É possível repetir `--lang` e `--category` para processar múltiplos valores. Para monitorar o progresso use:

```bash
python cli.py monitor
```

Também é possível enfileirar execuções futuras:

```bash
python cli.py queue --lang en --category "Algorithms"
```

### Normalização de categorias e busca automática

Os nomes de categoria passam por um processo de normalização que remove
acentos e converte para minúsculas. Isso permite utilizar aliases sem
preocupação com variações de escrita. Caso a categoria informada não exista, o
scraper tenta localizá-la automaticamente na Wikipédia.

Exemplo de uso equivalentes:

```bash
python cli.py scrape --lang pt --category programacao --format json
python cli.py scrape --lang pt --category "Programação" --format json
```

Para listar os arquivos gerados e visualizar configurações chave use:

```bash
python cli.py status
```

### Cache

Selecione o backend de cache com `--cache-backend` (`file`, `sqlite` ou `redis`)
e defina o tempo de vida dos registros com `--cache-ttl` (segundos). Para
remover entradas expiradas execute:

```bash
python cli.py clear-cache
```

### Logs

Use `--log-level` para ajustar a verbosidade (`DEBUG`, `INFO`, `WARNING`, etc.)
e `--log-format` para escolher entre saída `text` (padrão) ou `json`.
Exemplo:

```bash
python cli.py --log-level DEBUG --log-format json scrape --lang pt --category "Programação"
```

### Paralelismo

Controle o número de threads e processos utilizados pelo scraper com as opções
`--max-threads` e `--max-processes`. Esses valores também podem ser definidos
pelas variáveis de ambiente `MAX_THREADS` e `MAX_PROCESSES`.

## API FastAPI

Inicie a API executando:

```bash
uvicorn api_app:app --reload
```

Envie uma requisição `POST /scrape` com um JSON contendo `lang`, `category` e `format` para gerar o dataset.

## Dashboard

Para acompanhar o progresso do scraper basta rodar:

```bash
python cli.py monitor
```

Essa interface lê `logs/progress.json` e exibe o total de páginas processadas, uso de CPU e memória, além dos clusters, tópicos e idiomas atuais.

## Filas e Workers

O módulo `task_queue.py` abstrai o uso de backends como RabbitMQ. Execute `worker.py`
em contêiner separado para processar as tarefas enviadas por
`DatasetBuilder.build_from_pages(use_queue=True)`.

Um `Dockerfile.worker` está disponível para criar a imagem do worker e a pasta
contém o exemplo `cluster.yaml` para configuração de múltiplos nós com Dask ou
Ray. Em ambientes Kubernetes, basta criar um `Deployment` apontando para essa
imagem e montar o arquivo de configuração se necessário.
