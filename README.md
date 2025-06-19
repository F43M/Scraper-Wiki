# Scraper-Wiki
Scraper para criacao de datasets para fine tuning e treinamento de modelos de inteligencia artificial

## Instalação

Use o arquivo `requirements.txt` para instalar as dependências:

```bash
pip install -r requirements.txt
```

## Uso via linha de comando

Execute o scraper diretamente especificando idiomas, categorias e formato de saída:

```bash
python scraper_wiki.py --lang pt --category "Programação" --format json
```

É possível repetir `--lang` e `--category` para processar múltiplos valores.

## API FastAPI

Inicie a API executando:

```bash
uvicorn api_app:app --reload
```

Envie uma requisição `POST /scrape` com um JSON contendo `lang`, `category` e `format` para gerar o dataset.
