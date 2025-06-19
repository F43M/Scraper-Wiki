# Scraper-Wiki
Scraper para criacao de datasets para fine tuning e treinamento de modelos de inteligencia artificial

## Instalação

Use o arquivo `requirements.txt` para instalar as dependências:

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

Para listar os arquivos gerados e visualizar configurações chave use:

```bash
python cli.py status
```

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
