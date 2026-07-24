# Data-Analyst Telegram Bot

An LLM agent that answers data-analysis questions sent to it on Telegram.
Built for IIT Madras Tools in Data Science, Project 1 — Q5.

## What it does

Message the bot a data-analysis question (inline data, or a pointer to a public
dataset such as MOSPI). The agent works out the answer — fetching data and
running pandas/numpy code in a sandboxed `run_python` tool when needed — and
replies with exactly one JSON object:

```json
{"answer": {"state": "Assam"}, "log_url": "https://<host>/run.jsonl"}
```

- `answer` is shaped exactly as the question asks.
- `log_url` is a public, wget-able JSONL log of every agent step.

Multi-turn conversations are supported: per-chat history is kept and the agent
answers the latest message in context.

## Architecture

```
FastAPI web app ──► GET /health        (keep-alive + sanity check)
                └─► GET /run.jsonl     (the public agent log)

Background thread ──► Telegram getUpdates long-poll loop
                      └─► per-message: agent loop → sendMessage(JSON)

Background thread ──► self-ping /health every 10 min (free hosts idle out)
```

## Setup

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `BOT_TOKEN` | Yes | Telegram bot token from @BotFather |
| `AIPIPE_TOKEN` | Yes | AI Pipe API token for LLM access |
| `MODEL` | No | Model name (default: `gpt-4o`) |
| `MODEL_BASE_URL` | No | OpenAI-compatible base URL (default: `https://aipipe.org/openai/v1`) |
| `BASE_URL` | Yes | Public URL of the deployed service (e.g. `https://your-bot.onrender.com`) |

### Run locally

```bash
pip install -r requirements.txt
export BOT_TOKEN="your-token"
export AIPIPE_TOKEN="your-token"
export BASE_URL="http://localhost:8000"
uvicorn bot:app --host 0.0.0.0 --port 8000
```

### Deploy on Render

1. Push to a **public** GitHub repo.
2. Create a Render **Web Service**:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
3. Set env vars: `BOT_TOKEN`, `AIPIPE_TOKEN`, `BASE_URL=https://<service>.onrender.com`
4. Trigger a deploy after changing env vars.

### Verify

```bash
curl https://<your-host>/health
wget https://<your-host>/run.jsonl
```
