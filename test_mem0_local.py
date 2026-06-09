#!/usr/bin/env python3
"""
Mem0 Local Integration Test
Verifies: Memory init, add, search, get_all

Run with Hermes' Python:
  /path/to/hermes/python test_mem0_local.py
"""

import json
import os
import sys


def main():
    api_key = os.environ.get("AGNES_API_KEY", "")
    base_url = os.environ.get("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1")
    model = os.environ.get("AGNES_MODEL", "agnes-2.0-flash")

    if not api_key:
        print("❌ AGNES_API_KEY not set")
        print("   export AGNES_API_KEY=***")
        sys.exit(1)

    print(f"LLM:     {model} @ {base_url}")
    print(f"API Key: {api_key[:8]}...")
    print()

    # ------------------------------------------------------------------ 
    # Build config
    # ------------------------------------------------------------------ 
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": model,
                "openai_base_url": base_url,
                "api_key": api_key,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "fastembed",
            "config": {"model": "BAAI/bge-small-zh-v1.5"},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": "/tmp/mem0_test_qdrant",
                "embedding_model_dims": 512,
            },
        },
        "version": "v1.1",
    }

    # ------------------------------------------------------------------ 
    # Create Memory
    # ------------------------------------------------------------------ 
    print("[1/4] Initializing Mem0 (library mode)...")
    from mem0 import Memory
    memory = Memory.from_config(config)
    print("  ✅ OK")
    print()

    # ------------------------------------------------------------------ 
    # Add
    # ------------------------------------------------------------------ 
    print("[2/4] Adding a test memory...")
    result = memory.add(
        [
            {"role": "user", "content": "I prefer dark mode in all my editors."},
            {"role": "assistant", "content": "Noted! Using dark mode."},
        ],
        user_id="test-user",
        agent_id="hermes",
    )
    memories = result.get("results", [])
    print(f"  ✅ Added {len(memories)} memories")
    for m in memories:
        print(f"     → {m.get('memory', '')}")
    print()

    # ------------------------------------------------------------------ 
    # Search
    # ------------------------------------------------------------------ 
    print("[3/4] Searching...")
    result = memory.search(
        "dark mode editor",
        filters={"user_id": "test-user", "agent_id": "hermes"},
        top_k=5,
    )
    results = result.get("results", [])
    print(f"  ✅ Found {len(results)} results")
    for r in results:
        print(f"     [{r.get('score', 0):.3f}] {r.get('memory', '')}")
    print()

    # ------------------------------------------------------------------ 
    # Get all
    # ------------------------------------------------------------------ 
    print("[4/4] Listing all memories...")
    result = memory.get_all(
        filters={"user_id": "test-user", "agent_id": "hermes"},
        top_k=100,
    )
    results = result.get("results", [])
    print(f"  ✅ Total: {len(results)} memories")
    for r in results:
        print(f"     · {r.get('memory', '')}")
    print()

    # ------------------------------------------------------------------ 
    # Cleanup
    # ------------------------------------------------------------------ 
    memory.close()
    import shutil
    shutil.rmtree("/tmp/mem0_test_qdrant", ignore_errors=True)

    print("All tests passed! 🎉")


if __name__ == "__main__":
    main()
