# Scraper-Wiki
Scraper para criacao de datasets para fine tuning e treinamento de modelos de inteligencia artificial

A documentação da API está disponível em [docs/_build/index.html](docs/_build/index.html).

Para gerar a documentação HTML utilize:

```bash
pdoc -o docs -d google integrations core plugins utils api
```


## Instalação

Instale as dependências com `poetry`:

O projeto requer **Python 3.11** ou superior.

```bash
poetry install
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
Para gerar pares pergunta/resposta use `--format qa`:

```bash
python cli.py scrape --lang pt --category "Programação" --format qa
```
Para salvar um corpus de texto simples utilize `--format text`:

```bash
python cli.py scrape --lang pt --category "Programação" --format text
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

Para limitar o número de tarefas executadas em paralelo pelo worker assíncrono
defina `WORKER_CONCURRENCY` (padrão `5`). O número máximo de requisições HTTP
em andamento continua controlado por `MAX_CONCURRENT_REQUESTS`.

Para acelerar a coleta também é possível ativar o modo assíncrono com `--async`,
que realiza múltiplas requisições HTTP em paralelo respeitando o limite definido
por `MAX_CONCURRENT_REQUESTS`.

### URLs base personalizadas

O pacote `scraper_wiki` define o dicionário `BASE_URLS` com os domínios
principais para cada idioma. A função `get_base_url(lang)` consulta esse mapa e
retorna `"https://{lang}.wikipedia.org"` quando o idioma não está definido.

### Proxies e User-Agents

Defina `Config.PROXIES` com uma lista de proxies rotativos e adicione strings
em `Config.USER_AGENTS` para alternar automaticamente o cabeçalho
``User-Agent`` a cada requisição.

### Armazenamento

Escolha onde salvar os datasets com `--storage-backend` ou variável `STORAGE_BACKEND`.
Os valores suportados são `local` (padrão), `s3`/`minio`, `mongodb`, `postgres`, `iceberg`, `neo4j` e `milvus`/`weaviate`.
Para S3/MinIO defina `S3_BUCKET` e `S3_ENDPOINT` (ou `MINIO_ENDPOINT`).
Para MongoDB use `MONGODB_URI`. Para PostgreSQL defina `POSTGRES_DSN`.
Para Apache Iceberg/Delta Lake defina `DATALAKE_PATH`. Para Neo4j utilize `NEO4J_URI`, `NEO4J_USER` e `NEO4J_PASSWORD`. Para bancos vetoriais use `MILVUS_URI` e `MILVUS_COLLECTION` ou `WEAVIATE_URI`.

O backend também pode ser escolhido definindo `STORAGE_BACKEND=<opção>`
no ambiente (por exemplo `STORAGE_BACKEND=postgres`). Cada backend possui
variáveis específicas para a conexão:

- **MongoDB**: configure `MONGODB_URI`, `MONGODB_DB` e
  `MONGODB_COLLECTION` (padrões `scraper` e `dataset`). A coleção deve
  existir e receber os documentos gerados.
- **PostgreSQL**: defina `POSTGRES_DSN` e `POSTGRES_TABLE` (padrão
  `dataset`). Crie previamente a tabela com o esquema abaixo:

```sql
CREATE TABLE dataset (
    id SERIAL PRIMARY KEY,
    data JSONB NOT NULL
);
```

Um DSN de exemplo é `dbname=scraper user=postgres password=secret host=localhost`.

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

Os plugins permitem estender o scraper com novos analisadores.
`plugins/` inclui `infobox_parser` e `table_parser`, que extraem
respectivamente infoboxes e tabelas das páginas da Wikipédia. Outros
plugins disponíveis são `api_docs`, `code_extractor`, `gist_scraper`,
`codepen_scraper`, `pdf_books`, `gitlab_scraper`, `gitlab_snippets`,
`competitions`, `bug_history_scraper`, `legacy_forums` e
`stackexchange` (acessível como `stackoverflow`). Use `--plugin` ou o
campo `plugin` na API para escolher qual utilizar.

O módulo `crawling.auto_learner` inclui ainda o `AutoLearnerScraper`, um
gerador de datasets que utiliza Selenium e `fake-useragent` para navegar por
páginas dinâmicas. Ele pode ser executado de forma independente:

```python
from core import AutoLearnerScraper
from crawling.auto_dataset import convert_records_to_dataset

scraper = AutoLearnerScraper("https://exemplo.com")
records = scraper.build_dataset(["python"])
dataset = convert_records_to_dataset(records, "pt", "Programação")
scraper.close()
```

### Otimizando workers assíncronos

Ao rodar o `AutoLearnerScraper` com Selenium é comum abrir diversos drivers em
paralelo. Utilize o parâmetro `max_workers` de forma conservadora e sempre em
modo *headless* para manter o consumo de memória baixo. Essa abordagem permite
que múltiplas páginas sejam processadas sem picos de CPU, mesmo em ambientes com
recursos limitados.

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

### Configuração via YAML

O arquivo `examples/code_dataset_config.yaml` demonstra como definir idiomas,
categorias e opções específicas de cada plugin em um único lugar. Para executar
o scraping carregando essa configuração utilize:

```python
import yaml
from plugins import load_plugin, run_plugin

cfg = yaml.safe_load(open("examples/code_dataset_config.yaml", encoding="utf-8"))
langs = cfg.get("languages", [])
cats = cfg.get("categories", [])
fmt = cfg.get("format", "json")
for name, opts in cfg.get("plugins", {}).items():
    plugin = load_plugin(name)(**opts)
    run_plugin(plugin, langs, cats, fmt)
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
uvicorn api.api_app:app --reload
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

### Checkpoints

Após cada atualização de progresso o scraper salva `checkpoint_pages.json` com as páginas pendentes e `checkpoint_data.json` com os registros já processados.
Se a execução for interrompida, esses arquivos serão carregados automaticamente na próxima inicialização e o processamento continuará de onde parou.

O projeto expõe métricas no formato Prometheus através da função `metrics.start_metrics_server()`. Estão disponíveis os contadores e o histograma:

- `scrape_success_total`
- `scrape_error_total`
- `scrape_block_total`
- `pages_scraped_total`
- `requests_failed_total`
- `request_retries_total`
- `page_processing_seconds`
- `scrape_session_seconds`
- `dataset_completeness_ratio`
- `dataset_topic_diversity`

Esses valores podem ser consultados por Prometheus e visualizados em dashboards Grafana para monitorar o scraping.

## Filas e Workers

O módulo `task_queue.py` abstrai o uso de backends de fila. Utilize `QUEUE_URL`
para apontar para instâncias `redis://` ou `amqp://` (RabbitMQ). Execute
`worker.py` ou `worker.py --async` em contêiner separado para processar as
tarefas enviadas por `DatasetBuilder.build_from_pages(use_queue=True)`.

O nível de paralelismo do worker assíncrono é controlado por
`WORKER_CONCURRENCY`.

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

## Scrapy

Para rastrear páginas em larga escala é possível usar o spider
`scraper_wiki.scrapy_spider.WikiSpider`. Execute com `scrapy runspider` e
defina `lang` e `category` se desejar iniciar por uma categoria específica:

```bash
scrapy runspider scraper_wiki/scrapy_spider.py -a lang=pt -a category=Programacao
```

## Spark Pipeline

O módulo `training.spark_pipeline` oferece uma etapa de pré-processamento
distribuído com PySpark. Basta informar o dataset de entrada e o diretório de
saída para gerar os pares pergunta/resposta em cluster:

```bash
python -m training.spark_pipeline dataset.json out_dir
```

## Controle de Qualidade

Antes de salvar os dados, o `DatasetBuilder` aplica três etapas de deduplicação
e validação:

1. `deduplicate_by_hash` remove entradas idênticas pelo hash do conteúdo;
2. `deduplicate_by_embedding` descarta registros muito semelhantes pelos embeddings;
3. `deduplicate_by_simhash` detecta textos quase idênticos usando Simhash.
4. Funções de *leak detection* verificam sobreposição com conjuntos de referência.

Em seguida, os registros passam por verificações de integridade dos campos e dos
embeddings para garantir consistência.

### Arquivo `dataset_info.json`

O método `save_dataset` gera também o arquivo `dataset_info.json` com
informações básicas sobre o conjunto criado:

- `source`: origem dos registros, por exemplo `"wikipedia"`;
- `collection_date`: data da coleta no formato `AAAA-MM-DD`;
- `license`: licença aplicável, padrão `"CC BY-SA 4.0"`.

### Publicação no Hugging Face

Utilize a função `publish_hf_dataset` em `training.formats` para enviar o
dataset diretamente para o Hub. Basta informar o repositório de destino e um
token válido:

```python
from training.formats import publish_hf_dataset
publish_hf_dataset(records, "usuario/meu-dataset", token="hf_xxx")
```

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

Utilize também os utilitários em `training.pretrained_utils`:

```python
from pathlib import Path
from training.pretrained_utils import prepare_bert_inputs, extract_image_dataset

# Tokenizar textos com BERT
inputs = prepare_bert_inputs(["exemplo de texto"])

# Gerar dataset de imagens para Stable Diffusion
records = [{"image_url": "http://site/img.jpg", "caption": "Uma foto"}]
extract_image_dataset(records, Path('sd_data'))
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

#### Exemplo de registro

```json
{
  "id": "123abc",
  "title": "Title",
  "language": "en",
  "category": "History",
  "topic": "computing",
  "subtopic": "pioneers",
  "keywords": ["Ada"],
  "tags": [],
  "content": "Ada worked at IBM.",
  "summary": "Ada summary",
  "context": "Ada summary",
  "content_embedding": [0.0],
  "summary_embedding": [0.0],
  "quality_score": 0.0,
  "tests": [],
  "questions": [{"text": "Who was Ada?"}],
  "answers": [{"text": "Ada worked at IBM."}],
  "relations": [],
  "docstring": "",
  "raw_code": "",
  "problems": [],
  "fixed_version": "",
  "lessons": "",
  "origin_metrics": {},
  "challenge": "",
  "images": [],
  "created_at": "2024-01-01T00:00:00",
  "metadata": {"source": "wikipedia", "length": 17}
}
```

### Fine-tuning de Modelos de Código

O plugin `github_scraper` permite coletar READMEs e arquivos de projetos no
GitHub. Esses textos podem ser usados para treinar modelos de geração de
código, como os baseados em Transformers. Um fluxo simples é:

```python
from plugins import load_plugin, run_plugin
from datasets import Dataset

plugin = load_plugin("github_scraper")
records = run_plugin(plugin, ["en"], ["machine-learning"])
ds = Dataset.from_list(records)
ds.save_to_disk("github_code")
```

O diretório salvo pode ser carregado por `datasets` e utilizado em
`examples/code_fine_tuning.ipynb`, que demonstra o fine-tuning do modelo
`codegen-350M-multi` utilizando o `Trainer` da biblioteca Transformers.

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
para treinamento de modelos. Para executá-los, instale as dependências com `poetry` e
abra o Jupyter:

```bash
jupyter notebook examples/ner_training.ipynb
```

O arquivo `ner_training.ipynb` carrega o dataset via Hugging Face e realiza
um treinamento rápido de NER com Transformers. Outros notebooks disponíveis
incluem `qa_training.ipynb`, que mostra como filtrar por tópico e idioma para
fazer fine-tuning de um modelo de Perguntas e Respostas, e `filtering.ipynb`,
que demonstra como selecionar artigos por categoria ou linguagem e exportar
subconjuntos do dataset. Novos notebooks `jax_fine_tuning.ipynb` e
`lightning_fine_tuning.ipynb` apresentam exemplos de treino usando JAX/Flax e
PyTorch Lightning.

## Roadmap

As próximas etapas previstas para o Scraper Wiki incluem:

- Integração de novos backends de armazenamento vetorial;
- Melhorias na interface web de monitoramento;
- Plugins adicionais para fontes de dados governamentais;
- Exportação simplificada para formatos usados em modelagem de linguagem;
- Otimizações de desempenho para grandes volumes de URLs.

