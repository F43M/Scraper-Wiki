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
Para gerar um arquivo no formato JSON Lines basta usar `--format jsonl`:

```bash
python cli.py scrape --lang pt --category "Programação" --format jsonl
```
Para gravar em TFRecord basta definir `--format tfrecord`:

```bash
python cli.py scrape --lang pt --category "Programação" --format tfrecord
```

É possível repetir `--lang` e `--category` para processar múltiplos valores. Para monitorar o progresso use:

```bash
python cli.py monitor
```

Também é possível enfileirar execuções futuras:

```bash
python cli.py queue --lang en --category "Algorithms"
```

### Obter HTML de uma ou mais páginas

O script `main.py` fornece uma interface avançada baseada em **Click**. Ele
permite escolher o formato de saída (`json`, `jsonl`, `csv` ou `parquet`) e também ler
uma lista de URLs de um arquivo para processamento em lote.

```bash
python main.py --url "https://en.wikipedia.org/wiki/Python" --output csv

# Para modo batch
python main.py --file urls.txt --output parquet
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

Para acelerar a coleta também é possível ativar o modo assíncrono com `--async`,
que realiza múltiplas requisições HTTP em paralelo respeitando o limite definido
por `MAX_CONCURRENT_REQUESTS`.

### URLs base personalizadas

O módulo `scraper_wiki.py` define o dicionário `BASE_URLS` com os domínios
principais para cada idioma. A função `get_base_url(lang)` consulta esse mapa e
retorna `"https://{lang}.wikipedia.org"` quando o idioma não está definido.

### Armazenamento

Escolha onde salvar os datasets com `--storage-backend` ou variável `STORAGE_BACKEND`.
Os valores suportados são `local` (padrão), `s3`/`minio`, `mongodb` e `postgres`.
Para S3/MinIO defina `S3_BUCKET` e `S3_ENDPOINT` (ou `MINIO_ENDPOINT`).
Para MongoDB use `MONGODB_URI`. Para PostgreSQL defina `POSTGRES_DSN`.

### Utilidades de texto

O pacote `utils.text` oferece funções auxiliares:

- `clean_text` remove referências numéricas e espaços extras;
- `normalize_person` simplifica infoboxes de pessoas;
- `extract_entities` usa spaCy para listar entidades nomeadas.
- `parse_date` converte datas para o formato ISO 8601;
- `normalize_infobox` padroniza chaves e valores de infoboxes.
- `advanced_clean_text` elimina HTML e pode remover stopwords quando
  `Config.REMOVE_STOPWORDS` (ou variável `REMOVE_STOPWORDS=1`) está ativado.
- O módulo `utils.cleaner` oferece `clean_wiki_text` para remover links, templates e tags HTML
  e `split_sentences` que divide o texto em frases usando spaCy ou NLTK.

### Sistema de Plugins

Os plugins permitem estender o scraper com novos analisadores. Em
`plugins/` foram adicionados `infobox_parser` e `table_parser`, que
extraem respectivamente infoboxes e tabelas das páginas da Wikipédia.
Use `--plugin` ou o campo `plugin` na API para escolher qual utilizar.

Exemplo executando o plugin do StackOverflow:

```python
from plugins import load_plugin, run_plugin

plg = load_plugin("stackoverflow")
records = run_plugin(plg, ["en"], ["python"])
```

E para consultar itens da Wikidata:

```python
plg = load_plugin("wikidata")
records = run_plugin(plg, ["en"], ["Artificial intelligence"])
```

## Limpeza e NLP

Estas funções podem ser utilizadas isoladamente ou combinadas com o
`DatasetBuilder` e a API. Elas servem para higienizar o texto e extrair
informações estruturadas.

```python
from utils.text import (
    clean_text,
    normalize_person,
    normalize_infobox,
    parse_date,
    extract_entities,
)
from scraper_wiki import DatasetBuilder

# Processando uma página manualmente
builder = DatasetBuilder()
record = builder.process_page({"title": "Guido van Rossum", "lang": "en"})

# O texto já é limpo internamente, mas pode ser tratado novamente
cleaned = clean_text(record["content"])
entities = extract_entities(cleaned)
person = normalize_person({"name": record["title"], "occupation": "Programmer|BDFL"})
normalized = normalize_infobox({"title": record["title"], "date": "Jan 1, 1990"})
iso_date = parse_date("1 January 1990")
```

```python
# Pós-processando registros vindos da API
import requests
from utils.text import clean_text, extract_entities

dataset = requests.get("http://localhost:8000/records").json()
first = dataset[0]
first["entities"] = extract_entities(clean_text(first["content"]))
```

## API FastAPI

Inicie a API executando:

```bash
uvicorn api_app:app --reload
```

Envie uma requisição `POST /scrape` com um JSON contendo `lang`, `category` e `format` para gerar o dataset.

### Consulta de registros

Os dados gerados podem ser recuperados via `GET /records` com filtros opcionais:

```bash
curl "http://localhost:8000/records?lang=pt&category=Programação"
```

Para consultas mais flexíveis existe o endpoint `POST /graphql` que aceita
consultas GraphQL usando `graphene`. Exemplo:

```bash
curl -X POST http://localhost:8000/graphql -H "Content-Type: application/json" \
  -d '{"query": "{ records(lang:[\"pt\"]) { title category } }"}'
```

Informações de progresso podem ser obtidas em `GET /stats`.

## Dashboard

Para acompanhar o progresso do scraper basta rodar:

```bash
python cli.py monitor
```

Essa interface lê `logs/progress.json` e exibe o total de páginas processadas, uso de CPU e memória, além dos clusters, tópicos e idiomas atuais.
Agora o dashboard também consulta `GET /stats` quando disponível para mostrar as estatísticas em tempo real.
Além das contagens, ele exibe a média de tempo de processamento das páginas baseada no histograma `page_processing_seconds`.

O projeto expõe métricas no formato Prometheus através da função `metrics.start_metrics_server()`. Estão disponíveis os contadores e o histograma:

- `scrape_success_total`
- `scrape_error_total`
- `scrape_block_total`
- `pages_scraped_total`
- `requests_failed_total`
- `request_retries_total`
- `page_processing_seconds`
- `scrape_session_seconds`

Esses valores podem ser consultados por Prometheus e visualizados em dashboards Grafana para monitorar o scraping.

## Filas e Workers

O módulo `task_queue.py` abstrai o uso de backends como RabbitMQ. Execute `worker.py`
em contêiner separado para processar as tarefas enviadas por
`DatasetBuilder.build_from_pages(use_queue=True)`.

Um `Dockerfile.worker` está disponível para criar a imagem do worker e a pasta
contém o exemplo `cluster.yaml` para configuração de múltiplos nós com Dask ou
Ray. Em ambientes Kubernetes, basta criar um `Deployment` apontando para essa
imagem e montar o arquivo de configuração se necessário.

## Execução distribuída

Para processar páginas em um cluster, defina um arquivo `cluster.yaml` como:

```yaml
cluster:
  backend: dask  # ou 'ray'
  scheduler: tcp://scheduler:8786
```

Em seguida rode o comando com `--distributed`:

```bash
python cli.py scrape --distributed --lang pt --category "Programação"
```

O `DatasetBuilder` enviará as tarefas para o cluster usando `client.submit`.

## Controle de Qualidade

Antes de salvar os dados, o `DatasetBuilder` aplica três etapas de deduplicação
e validação:

1. `deduplicate_by_hash` remove entradas idênticas pelo hash do conteúdo;
2. `deduplicate_by_embedding` descarta registros muito semelhantes pelos embeddings;
3. `deduplicate_by_simhash` detecta textos quase idênticos usando Simhash.

Em seguida, os registros passam por verificações de integridade dos campos e dos
embeddings para garantir consistência.

## Integração com frameworks de ML

Os arquivos gerados em `training/` permitem treinar modelos de NLP de forma simples. A seguir alguns exemplos.

### Usando Transformers (PyTorch)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json, torch

pairs = json.load(open('datasets_wikipedia_pro/wikipedia_qa_pairs.json'))
model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
tokenizer = AutoTokenizer.from_pretrained('t5-small')
inputs = tokenizer(pairs[0]['question'], return_tensors='pt')
with torch.no_grad():
    output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Carregando embeddings com TensorFlow

```python
import json
import tensorflow as tf

emb = json.load(open('datasets_wikipedia_pro/wikipedia_qa_embeddings.json'))
emb_tensor = tf.constant([e['embedding'] for e in emb])
print(emb_tensor.shape)
```

Também é possível abrir o dataset salvo em TFRecord:

```python
import tensorflow as tf

dataset = tf.data.TFRecordDataset('datasets_wikipedia_pro/wikipedia_qa.tfrecord')
for raw in dataset.take(1):
    print(raw.numpy())
```

### Gerar grafo de conhecimento

```python
from utils.relation import relations_to_graph
from scraper_wiki import DatasetBuilder
import networkx as nx

builder = DatasetBuilder()
data = builder.generate_qa_pairs('Title', 'Ada worked at IBM.', 'Ada summary', 'en', 'History')
G = relations_to_graph(data['relations'])
nx.write_graphml(G, 'relations.graphml')
```

## Docker

Para executar a API e o worker em contêineres, primeiro construa a imagem base:

```bash
docker build -t scraper-api .
```

Em seguida utilize o `docker-compose.yml` para subir os serviços (API, worker e opcionalmente RabbitMQ):

```bash
docker-compose up
```

As imagens podem ser publicadas em um registro e implantadas em plataformas como Kubernetes ou AWS ECS para execução em escala.


## Notebooks de Exemplo

Os notebooks ficam em `examples/` e demonstram como utilizar os dados
para treinamento de modelos. Para executá-los, instale as dependências e
abra o Jupyter:

```bash
pip install -r requirements.txt jupyter
jupyter notebook examples/ner_training.ipynb
```

O arquivo `ner_training.ipynb` carrega o dataset via Hugging Face e realiza
um treinamento rápido de NER com Transformers.
