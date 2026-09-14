"""
The support agent: classify -> retrieve grounding -> draft reply -> decide
auto-handle vs escalate (with a stated reason). This is the module the three
assignment requirements map onto directly.

Run standalone: `python3 src/agent.py "some customer tweet text"`
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, asdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from intents import rule_based_intent, INTENTS
from retrieval import ExchangeIndex, RetrievalHit
from load_data import load_tweets, exchanges_for_brand, Exchange
import llm_client

BRAND = "AppleSupport"

# Escalation thresholds and rules -- these are documented, inspectable
# constants, not black-box model weights, because requirement #3 asks for a
# *stated* reason, and a stated reason has to trace back to something a
# human reviewer can check.
GROUNDING_SCORE_THRESHOLD = 0.12
FRUSTRATION_WORDS = {
    "fuck", "fucking", "shit", "wtf", "disgrace", "paralysed", "paralyzed",
    "hate", "worst", "grrrr", "grrrrrrrrrr", "beg",
}
FRUSTRATION_EMOJI = {"😡", "😠", "🤬"}
INTENTS_REQUIRING_IDENTITY = {"account_access"}
ALWAYS_AUTO_INTENTS = {"positive_feedback"}


@dataclass
class AgentOutput:
    text: str
    intent: str
    intent_reason: str
    reply_draft: str
    grounded_on: list[str]  # historical tweet_ids used for grounding
    top_grounding_score: float
    decision: str  # "auto_handle" | "escalate"
    decision_reason: str
    llm_backend: str


def frustration_score(text: str) -> int:
    lowered = text.lower()
    score = sum(1 for w in FRUSTRATION_WORDS if w in lowered)
    score += sum(1 for e in FRUSTRATION_EMOJI if e in text)
    score += text.count("!") >= 2
    score += len(re.findall(r"\b[A-Z]{4,}\b", text)) > 0  # shouting in caps
    return score


def draft_reply(customer_text: str, hits: list[RetrievalHit]) -> tuple[str, str]:
    """
    Returns (draft, backend_used).
    Grounding strategy: give the LLM the top retrieved historical exchanges
    as few-shot exemplars of how this brand actually resolves similar
    issues, and ask it to produce a reply in that voice for the new message.
    Offline fallback: adapt the single best-matching historical reply by
    swapping in the acknowledgement clause, which reproduces the brand's
    real observed pattern (acknowledge -> ask one diagnostic question ->
    invite to DM) without needing an LLM at all.
    """
    exemplars = "\n".join(
        f"- Customer said: \"{h.exchange.customer_tweet.text}\"\n"
        f"  Brand replied: \"{h.exchange.brand_tweet.text}\""
        for h in hits
    )
    system = (
        f"You are drafting a public Twitter reply as {BRAND}'s support team. "
        "Match the brand's real historical tone and structure shown in the "
        "examples: brief acknowledgement, one specific diagnostic question, "
        "invite the customer to continue in DM. Do not promise a fix or "
        "timeline you cannot know. Keep it under 220 characters."
    )
    user = (
        f"Historical examples of how {BRAND} has resolved similar issues:\n"
        f"{exemplars}\n\n"
        f"New customer message: \"{customer_text}\"\n\n"
        "Draft the reply."
    )
    backend = llm_client.backend_name()
    if backend != "offline-template":
        text, backend = llm_client.complete(system, user)
        if not text.startswith("[offline-template"):
            return text, backend

    # Offline fallback: reuse the structure of the best historical reply.
    if hits:
        best = hits[0].exchange.brand_tweet.text
        # Strip the original @handle mention so it reads as a fresh reply.
        best = re.sub(r"^@\S+\s*", "", best)
        return f"@customer {best}", "offline-template"
    return (
        "@customer Thanks for reaching out -- we'd like to help. "
        "Please DM us with more detail so we can look into this.",
        "offline-template",
    )


def decide(intent: str, customer_text: str, top_score: float) -> tuple[str, str]:
    if intent in ALWAYS_AUTO_INTENTS:
        return "auto_handle", "positive/resolved message -- safe to auto-acknowledge, no action required"

    if intent in INTENTS_REQUIRING_IDENTITY:
        return "escalate", "intent involves account/identity verification (e.g. access codes); auto-handling risks account-security mistakes, always route to a human"

    if intent == "other":
        return "escalate", "message did not match any defined intent with confidence; unclassified traffic defaults to human review"

    fscore = frustration_score(customer_text)
    if fscore >= 2:
        return "escalate", f"high frustration signal detected (score={fscore}); tone/retention risk is better judged by a human agent"

    if top_score < GROUNDING_SCORE_THRESHOLD:
        return "escalate", f"no sufficiently similar historical resolution found (top grounding score {top_score:.2f} < {GROUNDING_SCORE_THRESHOLD}); insufficient evidence to auto-draft confidently"

    return "auto_handle", f"intent '{intent}' has an established historical resolution pattern (grounding score {top_score:.2f}); safe to auto-send diagnostic follow-up"


def run_agent(customer_text: str, index: ExchangeIndex) -> AgentOutput:
    intent, intent_reason = rule_based_intent(customer_text)
    hits = index.search(customer_text, k=3)
    top_score = hits[0].score if hits else 0.0

    if intent == "positive_feedback":
        # BUG FOUND DURING EVAL (see DECISION_LOG.md #9): the retrieval
        # corpus for this brand's sample contains zero "thanks / resolved"
        # exchanges (every captured historical reply is a troubleshooting
        # response to an open complaint), so grounding a "problem solved!"
        # message against it produced a diagnostic question -- i.e. the
        # agent asked a happy customer "when did this start happening?".
        # Positive feedback needs no diagnosis, so it's handled directly
        # instead of going through the retrieval-grounded drafting path.
        draft, backend = "@customer That's what we love to hear! Thanks for sticking with us.", "template-positive-feedback"
    else:
        draft, backend = draft_reply(customer_text, hits)

    decision, decision_reason = decide(intent, customer_text, top_score)
    return AgentOutput(
        text=customer_text,
        intent=intent,
        intent_reason=intent_reason,
        reply_draft=draft,
        grounded_on=[h.exchange.brand_tweet.tweet_id for h in hits],
        top_grounding_score=top_score,
        decision=decision,
        decision_reason=decision_reason,
        llm_backend=backend,
    )


def build_index(
    csv_path: str = "data/twcs_sample.csv",
    brand: str = BRAND,
    exclude_customer_tweet_ids: set[str] | None = None,
) -> ExchangeIndex:
    """
    exclude_customer_tweet_ids: used for leave-one-out evaluation. Several
    golden-set messages are themselves in the historical exchange corpus
    (the same tweet that was actually sent, with its real captured reply).
    Querying the index without excluding it would retrieve itself at
    similarity 1.0 and make grounding look trivially perfect -- see
    eval/run_eval.py and DECISION_LOG.md #8.
    """
    tweets = load_tweets(csv_path)
    exchanges = exchanges_for_brand(tweets, brand)
    if exclude_customer_tweet_ids:
        exchanges = [e for e in exchanges if e.customer_tweet.tweet_id not in exclude_customer_tweet_ids]
    return ExchangeIndex(exchanges)


if __name__ == "__main__":
    idx = build_index()
    text = sys.argv[1] if len(sys.argv) > 1 else "@AppleSupport my battery is draining so fast after the update, please help"
    out = run_agent(text, idx)
    import json
    print(json.dumps(asdict(out), indent=2))
