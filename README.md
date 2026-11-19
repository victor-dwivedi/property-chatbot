# DarGlobal & Wasalt Property Chatbot

RAG chatbot that answers questions about DarGlobal projects and Wasalt (Saudi Arabia) property listings,
using data scraped from their public websites and a **free LLM via OpenRouter**.

```
scraper/  ──►  data/*.jsonl  ──►  app/ (FastAPI + BM25 retrieval)  ──►  OpenRouter free model
                                         └── static chat UI at  /
```

| Part | What it does |
|---|---|
| `scraper/wasalt.py` | Pulls city-level category pages from Wasalt's public sitemap and extracts the embedded listing JSON (type, price, city, district, beds, baths, area, amenities, URL). **Drops all contact/personal data** (phones, WhatsApp, owner/agent names, licence numbers), crawls at a moderate rate, and keeps source URLs for the "Source: Wasalt.sa" attribution that [wasalt.sa/llms.txt](https://wasalt.sa/llms.txt) asks for. |
| `scraper/darglobal.py` | Reads DarGlobal's public sitemap and extracts project, landing and insight pages. Bot-challenge responses are skipped, not bypassed. |
| `app/main.py` | Loads `data/*.jsonl` and builds a BM25 index with boosts for city, sale/rent, bedrooms and budget ("under 2M"). It sends the top matches to OpenRouter with a grounded system prompt, falling back across free models. |
| `app/static/index.html` | Chat UI that works on mobile and in light or dark mode. |

## Run locally

```bash
pip install -r requirements-scraper.txt
python scraper/wasalt.py          # -> data/wasalt_listings.jsonl
python scraper/darglobal.py       # -> data/darglobal_pages.jsonl

docker build -t property-chatbot .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=sk-or-... property-chatbot
# open http://localhost:7860
```

Environment variables:
* `OPENROUTER_API_KEY` (required)
* `OPENROUTER_MODELS`: comma-separated fallback list. The default is `openrouter/free` followed by specific `:free` models.
* `TOP_K`: number of retrieved documents, default 12.

## Deploy (free)

**Hugging Face Spaces (Docker):** create a Space with the Docker SDK and push this repo to it. Under Settings → Variables and secrets, add `OPENROUTER_API_KEY`. The app listens on port 7860, which is the Spaces default.

**Render:** push to GitHub, then choose New → Blueprint and pick this repo. `render.yaml` is included. When prompted, set `OPENROUTER_API_KEY`.

## Endpoints
* `GET /`: chat UI
* `POST /api/chat`: request `{"message": "...", "history": [...]}`, response `{answer, model, sources[]}`
* `GET /health`: health check
* `GET /api/stats`: knowledge-base summary
