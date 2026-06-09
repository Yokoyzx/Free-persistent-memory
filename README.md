# Mem0 Local — Hermes Agent Memory Provider

A lightweight, fully-local memory backend for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Runs in-process — **no Docker, no external services**.

- **LLM Extraction** — [Agnes AI](https://agnes-ai.com/) (OpenAI-compatible) extracts facts from conversations
- **Chinese Embeddings** — `bge-small-zh-v1.5` via fastembed (30MB, 512d)
- **Vector Storage** — Qdrant on-disk (no server needed)
- **Zero Docker** — Everything runs inside Hermes' Python process

## How It Works

```
Hermes Agent
  └─ memory.provider = mem0_local
       └─ plugins/memory/mem0_local/__init__.py
            ├─ LLM:       Agnes AI (openai-compatible)
            ├─ Embedder:  fastembed + bge-small-zh (30MB, 512d)
            └─ Vector DB: Qdrant on-disk
```

## Prerequisites

- [Hermes Agent](https://hermes-agent.nousresearch.com/) installed
- An **Agnes API key** — free tier available at [agnes-ai.com](https://agnes-ai.com/)
- Internet connection (first run downloads ~30MB embedding model)

## Installation

### 1. Install Python Dependencies

```bash
# Locate Hermes' Python
HERMES_PYTHON=$(dirname $(which hermes))/python

# On Windows (git-bash):
# HERMES_PYTHON=/c/Users/<you>/AppData/Local/hermes/hermes-agent/venv/Scripts/python

# Install
$HERMES_PYTHON -m pip install mem0ai fastembed
```

> **Windows note:** If `pip` is missing, run `$HERMES_PYTHON -m ensurepip` first.

### 2. Configure Agnes API Key

Add these to your `~/.hermes/.env` (or `$HERMES_HOME/.env`):

```bash
AGNES_API_KEY=sk-your-key-here
AGNES_BASE_URL=https://apihub.agnes-ai.com/v1
AGNES_MODEL=agnes-2.0-flash
```

Get your key at [agnes-ai.com](https://agnes-ai.com/).

### 3. Create Config File

`~/.hermes/mem0_local.json`:

```json
{
  "embedder_model": "BAAI/bge-small-zh-v1.5",
  "user_id": "your-username",
  "agent_id": "hermes",
  "rerank": false
}
```

### 4. Install the Plugin

Copy the plugin to Hermes' plugin directory:

```bash
cp -r plugins/memory/mem0_local ~/.hermes/plugins/memory/
```

Or create `~/.hermes/plugins/memory/mem0_local/__init__.py` with content from [`plugins/memory/mem0_local/__init__.py`](plugins/memory/mem0_local/__init__.py).

### 5. Activate

```bash
hermes config set memory.provider mem0_local
```

### 6. Verify

```bash
# Start a test session
hermes chat -q "What do you remember about me?"

# Or run the test script
$HERMES_PYTHON test_mem0_local.py
```

## Memory Tools

Once active, Hermes gains these tools:

| Tool | Purpose |
|------|---------|
| `mem0_profile` | List all stored memories |
| `mem0_search` | Semantic search across memories |
| `mem0_conclude` | Store a fact verbatim |

Memories are automatically extracted from conversations and recalled before each turn.

## Customizing the LLM

You can swap the fact-extraction LLM to any OpenAI-compatible provider without changing the plugin. A few free / low-cost options:

| Provider | Model Example | Config |
|----------|--------------|--------|
| **[Agnes AI](https://agnes-ai.com/)** (default) | `agnes-2.0-flash` | `AGNES_API_KEY`, `AGNES_BASE_URL`, `AGNES_MODEL` |
| **Google Gemini** (free tier) | `gemini-2.0-flash` | [Get API key](https://aistudio.google.com/apikey) |
| **Groq** (free tier) | `mixtral-8x7b-32768` | [Get API key](https://console.groq.com/keys) |
| **DeepSeek** (cheap) | `deepseek-chat` | [Get API key](https://platform.deepseek.com/) |
| **OpenRouter** (multi-model) | `openai/gpt-4o-mini` | [Get API key](https://openrouter.ai/keys) |
| **Local Ollama** (fully offline) | `qwen2.5:7b` | [Install Ollama](https://ollama.com/) |

To switch, update `~/.hermes/.env`:

```bash
# Example: Google Gemini
GOOGLE_API_KEY=***
# No changes to the plugin needed — the plugin uses OpenAI-compatible config
```

Then update the plugin config in `~/.hermes/mem0_local.json` (or modify `_get_memory()` in the plugin if your provider needs a different `provider` field).

> **Note:** The `mem0_local` plugin is pre-configured with `"provider": "openai"` (OpenAI-compatible). Providers that share this format (Gemini via `openai_base_url`, Groq, DeepSeek, OpenRouter, Ollama) work out of the box. For Anthropic or Cohere, change `"provider"` to `"anthropic"` or `"cohere"` in the plugin's config dict.

## Reset Data

Delete the local database to start fresh:

```bash
rm -rf ~/.hermes/mem0_data/
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `model_not_found` / 503 | Model names are case-sensitive. Check `AGNES_MODEL` in `.env` |
| Dimension mismatch | Delete `~/.hermes/mem0_data/` and restart |
| Provider not showing | Run `hermes memory status` and start a **new** session |

## Files

| Path | Purpose |
|------|---------|
| `~/.hermes/.env` | Agnes API credentials |
| `~/.hermes/mem0_local.json` | Provider config |
| `~/.hermes/plugins/memory/mem0_local/` | Plugin code |
| `~/.hermes/mem0_data/` | Local vector database |

## Acknowledgments

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — The AI agent framework
- [Mem0](https://github.com/mem0ai/mem0) — Open-source memory layer (Apache 2.0)
- [Agnes AI](https://agnes-ai.com/) — LLM API provider
- [fastembed](https://github.com/qdrant/fastembed) — Lightweight embedding inference
- [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh-v1.5) — Chinese embedding model

## License

Apache 2.0
