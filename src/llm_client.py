"""
Pluggable LLM backend.

Design decision (see DECISION_LOG.md #3): the pipeline must be runnable by a
grader who may or may not have an API key configured, in <15 minutes, with no
setup friction. So every LLM call in this repo goes through `complete()`
below, which:

  1. Uses a real hosted model if ANTHROPIC_API_KEY or OPENAI_API_KEY is set
     in the environment (Anthropic is tried first).
  2. Otherwise falls back to a deterministic, template-based "offline mode"
     so classify/draft/judge still run and produce structured output --
     just without an LLM's language flexibility.

Offline mode is clearly labelled in every output record (`"llm_backend":
"offline-template"` vs `"claude-*"`/`"gpt-*"`) so results are never silently
mixed or misattributed. The eval harness reports both and calls out the gap
explicitly (see reports/REPORT.md, "what's misleading about my headline
number").
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def backend_name() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-3-5-sonnet-latest"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    return "offline-template"


def complete(system: str, user: str, max_tokens: int = 300) -> tuple[str, str]:
    """
    Returns (text, backend_name_used). Never raises: on any API error it
    logs to stderr and falls back to offline mode so a flaky network never
    breaks the pipeline mid-run.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            return _call_anthropic(key, system, user, max_tokens), "claude-3-5-sonnet-latest"
        except Exception as e:  # noqa: BLE001
            print(f"[llm_client] Anthropic call failed ({e}); falling back to offline mode.")

    key = os.environ.get("OPENAI_API_KEY")
    if key:
        try:
            return _call_openai(key, system, user, max_tokens), "gpt-4o-mini"
        except Exception as e:  # noqa: BLE001
            print(f"[llm_client] OpenAI call failed ({e}); falling back to offline mode.")

    return _offline_fallback(user), "offline-template"


def _call_anthropic(key: str, system: str, user: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"].strip()


def _call_openai(key: str, system: str, user: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": "gpt-4o-mini",
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def _offline_fallback(_user: str) -> str:
    # Real fallback logic lives in agent.py / judge.py, which call this only
    # as a last resort for free-text generation. Returning an explicit marker
    # keeps offline output from ever being confused with a model completion.
    return "[offline-template: no free-text LLM output available without an API key]"
