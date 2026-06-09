"""Mem0 Local memory plugin — MemoryProvider interface.

Library-mode Mem0 running in-process with local embeddings (fastembed + BGE)
and local vector storage (Qdrant on-disk). No external services needed.

Config via environment variables:
  AGNES_API_KEY       — Agnes API key (required)
  AGNES_BASE_URL      — Agnes API base URL (default: https://apihub.agnes-ai.com/v1)
  AGNES_MODEL         — Agnes model name (default: agnes-2.0-flash)

Or via $HERMES_HOME/mem0_local.json for non-secret settings.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars + JSON overrides.

    Priority: env vars > JSON file > defaults.
    """
    from hermes_constants import get_hermes_home

    config = {
        "agnes_api_key": os.environ.get("AGNES_API_KEY", ""),
        "agnes_base_url": os.environ.get("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1"),
        "agnes_model": os.environ.get("AGNES_MODEL", "agnes-2.0-flash"),
        "embedder_model": os.environ.get("MEM0_EMBEDDER_MODEL", "BAAI/bge-small-zh-v1.5"),
        "qdrant_path": "",
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": False,
    }

    config_path = get_hermes_home() / "mem0_local.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0LocalMemoryProvider(MemoryProvider):
    """Mem0 in-process library mode with local embeddings + Qdrant on-disk."""

    def __init__(self):
        self._config = None
        self._memory = None
        self._memory_lock = threading.Lock()
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0_local"

    def is_available(self) -> bool:
        cfg = _load_config()
        api_key = cfg.get("agnes_api_key", "")
        return bool(api_key)

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0_local.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0_local.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "agnes_api_key", "description": "Agnes API key", "secret": True, "required": True, "env_var": "AGNES_API_KEY"},
            {"key": "agnes_base_url", "description": "Agnes API base URL", "default": "https://apihub.agnes-ai.com/v1"},
            {"key": "agnes_model", "description": "Agnes model name", "default": "agnes-2.0-flash"},
            {"key": "embedder_model", "description": "Local embedding model", "default": "BAAI/bge-small-zh-v1.5"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
        ]

    def _get_memory(self):
        """Thread-safe lazy initializer for the Mem0 Memory object."""
        with self._memory_lock:
            if self._memory is not None:
                return self._memory
            try:
                from mem0 import Memory
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

            # Build config dict for Memory.from_config()
            config_dict = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": self._model_name,
                        "openai_base_url": self._base_url,
                        "api_key": self._api_key,
                        "max_tokens": 2000,
                    },
                },
                "embedder": {
                    "provider": "fastembed",
                    "config": {
                        "model": self._embedder_model,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": self._qdrant_path,
                        "embedding_model_dims": self._embedding_dims,
                    },
                },
                "version": "v1.1",
            }

            # Set history DB path
            if self._hermes_home:
                import os as _os
                mem0_dir = _os.path.join(self._hermes_home, "mem0_data")
                _os.makedirs(mem0_dir, exist_ok=True)
                config_dict["history_db_path"] = _os.path.join(mem0_dir, "history.db")

            try:
                self._memory = Memory.from_config(config_dict)
                logger.info("Mem0 Local initialized (model=%s, embedder=%s, qdrant=%s)",
                            self._model_name, self._embedder_model, self._qdrant_path)
                return self._memory
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Mem0 local: {e}")

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 Local circuit breaker tripped after %d consecutive failures.",
                self._consecutive_failures,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("agnes_api_key", "")
        self._base_url = self._config.get("agnes_base_url", "https://apihub.agnes-ai.com/v1")
        self._model_name = self._config.get("agnes_model", "agnes-2.0-flash")
        self._embedder_model = self._config.get("embedder_model", "BAAI/bge-small-zh-v1.5")
        # Determine embedding dimensions based on model
        model_dims = {
            "BAAI/bge-small-zh-v1.5": 512,
            "BAAI/bge-base-zh-v1.5": 768,
            "BAAI/bge-large-zh-v1.5": 1024,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-m3": 1024,
            "thenlper/gte-large": 1024,
            "thenlper/gte-base": 768,
            "thenlper/gte-small": 384,
        }
        self._embedding_dims = model_dims.get(self._embedder_model, 512)
        self._qdrant_path = self._config.get("qdrant_path", "")
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._hermes_home = kwargs.get("hermes_home", "")

        # Default Qdrant path if not set
        if not self._qdrant_path and self._hermes_home:
            import os as _os
            self._qdrant_path = _os.path.join(self._hermes_home, "mem0_data", "qdrant")
            _os.makedirs(self._qdrant_path, exist_ok=True)

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Local Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                memory = self._get_memory()
                results = memory.search(
                    query=query,
                    filters={"user_id": self._user_id, "agent_id": self._agent_id},
                    top_k=5,
                )
                if results and results.get("results"):
                    lines = [r.get("memory", "") for r in results["results"] if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 Local prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0l-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", **kwargs) -> None:
        if self._is_breaker_open():
            return

        def _sync():
            try:
                memory = self._get_memory()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                memory.add(
                    messages,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 Local sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0l-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 Local temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            memory = self._get_memory()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                results = memory.get_all(filters={"user_id": self._user_id, "agent_id": self._agent_id}, top_k=100)
                self._record_success()
                if not results or not results.get("results"):
                    return json.dumps({"result": "No memories stored yet."})
                lines = [r.get("memory", "") for r in results["results"] if r.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = memory.search(
                    query=query,
                    filters={"user_id": self._user_id, "agent_id": self._agent_id},
                    top_k=top_k,
                )
                self._record_success()
                if not results or not results.get("results"):
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)}
                         for r in results["results"]]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                memory.add(
                    [{"role": "user", "content": conclusion}],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._memory_lock:
            if self._memory:
                try:
                    self._memory.close()
                except Exception:
                    pass
                self._memory = None


def register(ctx) -> None:
    """Register Mem0 Local as a memory provider plugin."""
    ctx.register_memory_provider(Mem0LocalMemoryProvider())
