"""
Minimal Vercel-compatible demo API for the Hiver support agent.

Entrypoint: the `app` variable below is a Flask WSGI application that Vercel
discovers automatically (it looks for `app` in api/index.py, index.py, app.py,
etc.).

Design constraints honoured:
- Does NOT modify or import from the evaluation harness (eval/), golden set, or
  any report/decision-log file.
- Reuses src/agent.py logic directly — no duplication.
- Works fully in offline/deterministic mode with no API key required.
- The retrieval index is built ONCE at module load time (cold start) and reused
  across warm invocations — building it per-request would be ~200ms of wasted
  TF-IDF work on every call.
- All paths are resolved relative to this file's location so they work
  regardless of the calling process's cwd (important for Vercel's serverless
  runtime where cwd == project root, but explicit is safer).

Routes:
  GET  /api/           Health-check — returns {"status": "ok", ...}
  POST /api/predict    Run the agent on a customer message
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from flask import Flask, request, jsonify, Response

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# Resolve repo root from this file's location: api/index.py → repo root is ../
_HERE = Path(__file__).resolve().parent          # .../api/
_REPO_ROOT = _HERE.parent                        # .../hiver-support-agent/
_SRC = _REPO_ROOT / "src"
_DATA = _REPO_ROOT / "data" / "twcs_sample.csv"

# Add src/ to sys.path so agent.py can import its siblings (intents, retrieval…)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Now import the agent — these imports happen once at cold start.
from agent import run_agent, build_index  # noqa: E402  (after sys.path setup)

# ---------------------------------------------------------------------------
# Index — built once at cold start, reused across warm invocations.
# ---------------------------------------------------------------------------
_INDEX = build_index(csv_path=str(_DATA))

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/api/", methods=["GET"])
def health() -> Response:
    """Health-check endpoint. Returns dataset info so the caller can confirm
    the agent loaded correctly."""
    return jsonify({
        "status": "ok",
        "agent": "AppleSupport AI support agent",
        "mode": "offline-deterministic (no LLM API key required)",
        "historical_exchanges_indexed": len(_INDEX.exchanges),
        "data_file": "data/twcs_sample.csv (93-row sample)",
        "endpoints": {
            "GET /api/": "this health check",
            "POST /api/predict": "run agent on a customer message",
        },
    })


@app.route("/api/predict", methods=["POST"])
def predict() -> Response:
    """
    Run the support agent on a customer message.

    Request body (JSON):
        {"message": "@AppleSupport my battery is draining quickly"}

    Response body (JSON):
        {
          "intent":             "battery_life",
          "intent_reason":      "matched keyword rule for 'battery_life' (keyword: 'battery')",
          "top_grounding_score": 0.22,
          "grounded_on":        ["119298", "119293", "119300"],
          "decision":           "auto_handle",
          "decision_reason":    "intent 'battery_life' has an established ...",
          "draft_reply":        "@customer ...",
          "llm_backend":        "offline-template"
        }
    """
    # --- Parse request ---
    if not request.is_json:
        return jsonify({"error": "Request must be JSON with Content-Type: application/json"}), 415

    body = request.get_json(silent=True)
    if not body or "message" not in body:
        return jsonify({"error": "Request body must contain a 'message' field"}), 400

    message = body["message"]
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "'message' must be a non-empty string"}), 400

    # Hard cap to avoid absurdly large inputs hanging TF-IDF
    if len(message) > 1000:
        return jsonify({"error": "'message' must be 1000 characters or fewer"}), 400

    # --- Run agent ---
    try:
        out = run_agent(message.strip(), _INDEX)
    except Exception as exc:  # noqa: BLE001
        # Surface errors clearly rather than returning a 500 with no body
        return jsonify({"error": f"Agent error: {exc}"}), 500

    # --- Return only the fields the API consumer needs ---
    return jsonify({
        "intent":              out.intent,
        "intent_reason":       out.intent_reason,
        "top_grounding_score": round(out.top_grounding_score, 4),
        "grounded_on":         out.grounded_on,
        "decision":            out.decision,
        "decision_reason":     out.decision_reason,
        "draft_reply":         out.reply_draft,
        "llm_backend":         out.llm_backend,
    })


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run with: python api/index.py
    # The Flask dev server listens on http://127.0.0.1:5000
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
