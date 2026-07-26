"""Data-analyst Telegram bot — TDS Project 1, Q5.

An LLM agent that answers data-analysis questions sent over Telegram.
Replies to every message with exactly one JSON object:
    {"answer": <shaped as the question asks>, "log_url": "<public JSONL log>"}

Architecture:
  - FastAPI app serves /health and /run.jsonl (the public agent log).
  - A background thread long-polls Telegram getUpdates.
  - Each incoming message runs an agentic loop (OpenAI-compatible chat with a
    run_python tool) until the model produces the final JSON answer.
  - A keep-warm thread pings our own public URL so the free host never idles out.
"""

import io
import json
import os
import re
import threading
import time
import traceback
import contextlib
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

# ------------------------------------------------------------------ config
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
MODEL = os.environ.get("MODEL", "gpt-4o")
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "https://aipipe.org/openai/v1")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
LOG_PATH = os.environ.get("LOG_PATH", "/tmp/run.jsonl")
LOG_URL = f"{BASE_URL}/run.jsonl"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

MAX_AGENT_STEPS = 10
PY_TIMEOUT = 60        # seconds for one run_python call
ANSWER_BUDGET = 210     # wall-clock seconds before we force a final answer
KEEP_WARM_INTERVAL = 600  # 10 minutes

_log_lock = threading.Lock()
_histories: dict[int, list[dict]] = {}   # chat_id -> chat-completion messages
_hist_lock = threading.Lock()


# ------------------------------------------------------------------ FastAPI
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "base_url": BASE_URL}


@app.get("/run.jsonl")
def serve_log():
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    return PlainTextResponse(content, media_type="application/x-jsonlines")


# ------------------------------------------------------------------ logging
def log_event(**fields):
    """Append one JSON line to the JSONL log file."""
    fields["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(fields, ensure_ascii=False, default=str)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------ tools
def run_python(code: str) -> str:
    """Execute Python code, return captured stdout (or the error)."""
    out = io.StringIO()
    result: dict = {}

    def target():
        env = {"__name__": "__main__"}
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(code, env)
            result["ok"] = True
        except Exception:
            result["ok"] = False
            out.write("\n" + traceback.format_exc(limit=4))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(PY_TIMEOUT)
    if t.is_alive():
        return "ERROR: code timed out after %ss" % PY_TIMEOUT
    text = out.getvalue()
    return text[-8000:] if text else "(no output — use print())"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code on the server and get its printed output. "
                "pandas, numpy, requests, bs4, openpyxl, lxml are installed "
                "and the network is available (download public datasets with "
                "requests). Always print() what you need to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to execute",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """\
You are an expert data-analyst agent answering questions sent to a Telegram bot.

Rules:
1. Work out the answer to the user's LATEST message. Earlier messages in the \
chat are context for multi-turn tasks.
2. The message may embed data inline, or reference a public dataset \
(MOSPI, data.gov.in, SRS reports, etc.). Use the run_python tool to fetch \
data and compute — do not guess numeric results you can compute. \
For well-known published statistics (e.g. "which state has the highest \
maternal mortality rate per MOSPI/SRS"), you may answer from reliable \
knowledge if fetching fails.
3. The message usually spells out the exact JSON shape it wants. \
Output ONLY the JSON object the question asks for — no prose, no markdown \
fences, no explanation. Put "__LOG_URL__" as the log_url value; the system \
will substitute the real URL.
4. Match the requested `answer` shape exactly: same keys, same nesting, \
numbers as numbers (not strings) unless the question says string, etc. \
Never add extra keys.
5. If a mid-conversation message is only setup text ("I'll send data next", \
"Here is some data:", etc.) and does not ask a question, still reply with: \
{"answer": "ok", "log_url": "__LOG_URL__"}
6. NEVER wrap your JSON in markdown code fences or add any text before/after it.
7. If the question asks for a single value, return it directly in the answer \
field (string or number), not wrapped in an extra object unless the question \
explicitly shows an object shape.
8. When downloading data, try multiple approaches if the first fails. \
If all download attempts fail for a well-known statistic, answer from your \
knowledge rather than giving up.
"""


# ------------------------------------------------------------------ JSON extraction
def extract_json(text: str) -> dict:
    """
    Pull the first valid JSON object out of the model's response.
    Handles markdown fences, prose around JSON, nested braces.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Try the whole string first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Find first balanced { ... }
    start = text.find("{")
    if start == -1:
        return {"answer": text.strip()}

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                break

    # Last resort: try to find any JSON-like substring
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return {"answer": text.strip()}


def finalize_response(raw_text: str) -> str:
    """Extract JSON from model output, ensure answer key, set log_url."""
    obj = extract_json(raw_text)

    # Ensure "answer" key exists
    if "answer" not in obj:
        # If the object looks like it IS the answer shape, wrap it
        obj = {"answer": obj}

    # Always overwrite log_url with our real URL
    obj["log_url"] = LOG_URL
    return json.dumps(obj, ensure_ascii=False)


# ------------------------------------------------------------------ LLM calls
def chat_completion(messages: list[dict], tools=None) -> dict:
    """Call the OpenAI-compatible chat completion endpoint."""
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    resp = requests.post(
        f"{MODEL_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------ agent loop
def agent_loop(chat_id: int, user_message: str) -> str:
    """
    Run the agentic loop for one user message.
    Returns the final JSON string to send back.
    """
    deadline = time.time() + ANSWER_BUDGET

    # Get or create per-chat history
    with _hist_lock:
        if chat_id not in _histories:
            _histories[chat_id] = []
        history = _histories[chat_id]

    # Add user message to history
    history.append({"role": "user", "content": user_message})

    # Keep only the last 20 turns (40 messages: user+assistant pairs)
    if len(history) > 40:
        history[:] = history[-40:]

    # Build messages for the API call
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    log_event(event="agent_start", chat_id=chat_id, user_message=user_message)

    for step in range(MAX_AGENT_STEPS):
        remaining = deadline - time.time()
        if remaining <= 0:
            # Time's up — force an answer
            log_event(event="timeout_force", chat_id=chat_id, step=step)
            messages.append({
                "role": "user",
                "content": (
                    "TIME IS UP. You must answer RIGHT NOW with the JSON object. "
                    "Do NOT call any tools. Output only the JSON."
                ),
            })
            try:
                resp = chat_completion(messages, tools=None)
                raw = resp["choices"][0]["message"]["content"] or ""
            except Exception as e:
                raw = json.dumps({"answer": "timeout", "log_url": "__LOG_URL__"})
                log_event(event="timeout_error", error=str(e))
            break

        # Disable tools if running low on time (< 30s)
        use_tools = TOOLS if remaining > 30 else None

        try:
            resp = chat_completion(messages, tools=use_tools)
        except Exception as e:
            log_event(event="llm_error", chat_id=chat_id, step=step, error=str(e))
            raw = json.dumps({"answer": "llm_error", "log_url": "__LOG_URL__"})
            break

        choice = resp["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "")

        # If the model wants to call tools
        if msg.get("tool_calls"):
            # Append the assistant message with tool_calls
            messages.append(msg)

            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args_raw = tc["function"]["arguments"]
                tc_id = tc["id"]

                log_event(
                    event="tool_call",
                    chat_id=chat_id,
                    step=step,
                    tool=fn_name,
                    arguments=fn_args_raw,
                )

                if fn_name == "run_python":
                    try:
                        args = json.loads(fn_args_raw)
                        code = args.get("code", "")
                    except json.JSONDecodeError:
                        code = fn_args_raw

                    output = run_python(code)
                    log_event(
                        event="tool_result",
                        chat_id=chat_id,
                        step=step,
                        tool=fn_name,
                        output=output[:2000],
                    )
                else:
                    output = f"Unknown tool: {fn_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": output,
                })

            continue  # next iteration of the agent loop

        # Model produced a text response (no tool calls)
        raw = msg.get("content", "") or ""
        log_event(event="llm_response", chat_id=chat_id, step=step, content=raw[:2000])
        break
    else:
        # Exhausted all steps — force answer
        log_event(event="max_steps_reached", chat_id=chat_id)
        messages.append({
            "role": "user",
            "content": (
                "You have used all available tool steps. "
                "Answer NOW with the JSON object. No tools."
            ),
        })
        try:
            resp = chat_completion(messages, tools=None)
            raw = resp["choices"][0]["message"]["content"] or ""
        except Exception as e:
            raw = json.dumps({"answer": "max_steps_error", "log_url": "__LOG_URL__"})
            log_event(event="max_steps_error", error=str(e))

    # Finalize the response
    final = finalize_response(raw)

    # Store assistant reply in history
    history.append({"role": "assistant", "content": final})

    log_event(event="final_answer", chat_id=chat_id, answer=final)
    return final


# ------------------------------------------------------------------ Telegram
def tg_request(method: str, **kwargs):
    """Make a Telegram Bot API request."""
    url = f"{TG_API}/{method}"
    resp = requests.post(url, json=kwargs, timeout=60)
    return resp.json()


def send_message(chat_id: int, text: str):
    """Send a text message via Telegram."""
    return tg_request("sendMessage", chat_id=chat_id, text=text)


# ------------------------------------------------------------------ webhook endpoint
def _handle_message(chat_id: int, user_text: str):
    """Process a message in the background."""
    try:
        reply = agent_loop(chat_id, user_text)
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        log_event(event="handler_crash", chat_id=chat_id, error=str(e), traceback=tb)
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})
    send_message(chat_id, reply)
    print(f"[BOT] Replied to {chat_id}: {reply[:200]}", flush=True)


@app.post("/webhook")
async def webhook(request_data: dict):
    """Receive Telegram updates via webhook."""
    msg = request_data.get("message")
    if not msg or "text" not in msg:
        return {"ok": True}

    chat_id = msg["chat"]["id"]
    user_text = msg["text"]
    print(f"[BOT] Message from {chat_id}: {user_text[:100]}...", flush=True)

    # Process in background so Telegram gets instant 200 OK
    threading.Thread(target=_handle_message, args=(chat_id, user_text), daemon=True).start()
    return {"ok": True}


# ------------------------------------------------------------------ startup
@app.on_event("startup")
def on_startup():
    """Register Telegram webhook on boot."""
    # Ensure log file exists
    os.makedirs(os.path.dirname(LOG_PATH) if os.path.dirname(LOG_PATH) else ".", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            pass

    log_event(event="startup", model=MODEL, base_url=BASE_URL)

    # Register webhook with Telegram
    webhook_url = f"{BASE_URL}/webhook"
    result = tg_request("setWebhook", url=webhook_url)
    print(f"[STARTUP] setWebhook({webhook_url}) -> {result}", flush=True)

    print(f"[STARTUP] Bot is live! LOG_URL={LOG_URL}", flush=True)
