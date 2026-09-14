"""
Reply-quality judge: LLM-as-judge when an API key is available, deterministic
rubric-based scorer otherwise (same pluggable-backend pattern as agent.py --
see llm_client.py).

Rubric (1-5 each, mirrors what a human reviewer was asked to check in
eval/human_judge_scores.csv):
  - grounded:      does the reply reference something specific to the
                    customer's stated problem, rather than a pure generic?
  - actionable:    does it move the conversation forward (asks a concrete
                    diagnostic question or gives a concrete next step)?
  - tone:          is the tone appropriate (acknowledges the issue, not
                    dismissive, not over-promising)?
  - safe:          does it avoid fabricating a fix, a policy, or a promise
                    the brand can't actually guarantee?

Overall score = mean of the four. See eval/judge_agreement.py for how well
this offline rubric scorer agrees with a human on the same replies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

import llm_client

_HEDGE_WORDS = ["will fix", "guaranteed", "promise", "definitely will", "100%"]


@dataclass
class JudgeScore:
    grounded: int
    actionable: int
    tone: int
    safe: int
    overall: float
    backend: str
    rationale: str


def judge_reply(customer_text: str, reply: str) -> JudgeScore:
    backend = llm_client.backend_name()
    if backend != "offline-template":
        return _llm_judge(customer_text, reply, backend)
    return _rubric_judge(customer_text, reply)


def _llm_judge(customer_text: str, reply: str, backend: str) -> JudgeScore:
    system = (
        "You are grading a customer-support reply for AppleSupport on four "
        "1-5 axes: grounded (specific to the customer's actual problem), "
        "actionable (asks a concrete next step / diagnostic question), tone "
        "(acknowledges the issue without being dismissive or robotic), and "
        "safe (does not promise a fix, refund, or timeline it cannot know). "
        'Return ONLY compact JSON: {"grounded":int,"actionable":int,"tone":int,"safe":int,"rationale":str}'
    )
    user = f"Customer message: {customer_text}\nDrafted reply: {reply}"
    text, used_backend = llm_client.complete(system, user, max_tokens=200)
    try:
        data = json.loads(text)
        scores = [data["grounded"], data["actionable"], data["tone"], data["safe"]]
        return JudgeScore(
            grounded=data["grounded"], actionable=data["actionable"],
            tone=data["tone"], safe=data["safe"],
            overall=sum(scores) / 4, backend=used_backend,
            rationale=data.get("rationale", ""),
        )
    except Exception:  # noqa: BLE001
        # LLM returned something unparseable -- don't silently fabricate a
        # score, fall back to the rubric scorer and say so explicitly.
        fallback = _rubric_judge(customer_text, reply)
        fallback.rationale = "LLM judge output was unparseable; used rubric fallback. " + fallback.rationale
        return fallback


def _rubric_judge(customer_text: str, reply: str) -> JudgeScore:
    reply_l = reply.lower()
    customer_l = customer_text.lower()

    # grounded: does the reply share a content word with the customer message
    # (beyond stopwords), or reference iOS/device/DM in a way that's on-topic?
    content_words = set(re.findall(r"[a-z]{4,}", customer_l)) - {
        "applesupport", "please", "phone", "with", "that", "this", "have",
    }
    shared = [w for w in content_words if w in reply_l]
    grounded = 5 if len(shared) >= 2 else (4 if len(shared) == 1 else 2)

    # actionable: does it ask a question or explicitly invite DM / next step?
    actionable = 5 if ("?" in reply or "dm" in reply_l) else 2

    # tone: has an acknowledgement phrase, avoids curt/robotic one-liners
    ack_phrases = ["thanks", "sorry", "we'd like to help", "we can help", "that's what we love"]
    tone = 4 if any(p in reply_l for p in ack_phrases) else 3
    if len(reply) < 15:
        tone = min(tone, 2)  # too curt to read as acknowledging anything

    # safe: penalize over-promising language
    safe = 2 if any(h in reply_l for h in _HEDGE_WORDS) else 5

    overall = (grounded + actionable + tone + safe) / 4
    rationale = (
        f"offline rubric: shared_content_words={shared}, has_question_or_dm={'?' in reply or 'dm' in reply_l}, "
        f"has_ack_phrase={any(p in reply_l for p in ack_phrases)}, has_overpromise={safe == 2}"
    )
    return JudgeScore(grounded=grounded, actionable=actionable, tone=tone, safe=safe,
                       overall=overall, backend="offline-rubric", rationale=rationale)


if __name__ == "__main__":
    s = judge_reply(
        "my battery is draining so fast after the update",
        "@customer We'd like to help. Tell us more about the issue you're experiencing in DM.",
    )
    print(json.dumps(asdict(s), indent=2))
