"""
Intent taxonomy for @AppleSupport, defined by reading every customer message in
the provided sample that is directed at AppleSupport (17 messages -- see
eval/golden_set.csv for the full census). The taxonomy is intentionally small
(5 classes): the brief asks for "a small set of intents that you define from
the data", not exhaustive coverage of Apple's real support taxonomy.

INTENTS
-------
battery_life
    Battery draining faster than expected / degraded battery life, usually
    blamed on an iOS update. e.g. "drains it down 8 percent in 2 minutes".

update_performance
    General slowness, freezing, app crashes, or connectivity problems that
    the customer attributes to installing an iOS update. This is the
    dominant intent in the sample (8/17).

software_bug_ui
    A specific, reproducible bug/UI glitch not framed by the customer as an
    update side-effect (e.g. notification-under-keyboard bug).

account_access
    Account/App Store/iCloud verification, sign-in, or access-code problems.
    Distinct from device issues because resolving it requires touching the
    customer's account/identity, not just device diagnostics.

positive_feedback
    Problem already resolved / thanks / praise. No action needed other than
    acknowledgement.

Anything that doesn't fit is labelled "other" and always escalated (see
agent.py) -- an unclassified message is exactly the case where auto-handling
is riskiest.
"""
from __future__ import annotations

import re

from text_utils import strip_mentions_and_urls

INTENTS = [
    "battery_life",
    "update_performance",
    "software_bug_ui",
    "account_access",
    "positive_feedback",
    "other",
]

# Keyword/pattern rules used by the offline (no-LLM-key) classifier. Order
# matters: first matching rule wins. These were derived by reading the 17
# labelled examples in eval/golden_set.csv, not invented in the abstract.
_RULES: list[tuple[str, list[str]]] = [
    ("positive_feedback", ["problem solved", "love apple", "thanks", "thank you", "resolved"]),
    ("account_access", ["code", "verification", "i-store", "itunes", "icloud", "sign in", "sign-in", "password", "can't log", "cant log"]),
    ("battery_life", ["battery", "drain", "charge", "% ", "percent"]),
    ("software_bug_ui", ["notification", "keyboard", "bug", "glitch", "freez", "crash"]),
    ("update_performance", ["update", "ios", "slow", "wifi", "disconnect", "load", "whatsapp", "app"]),
]


def rule_based_intent(text: str) -> tuple[str, str]:
    """
    Deterministic fallback classifier (no LLM call). Returns (intent, reason).
    This is also Baseline 2 ("simple baseline") in the eval harness.

    Matches on word boundaries against mention/URL-stripped text. Both of
    these were fixed after an early version of this classifier tagged nearly
    every message as "update_performance" purely because the keyword "app"
    is a substring of "applesupport" -- i.e. the customer's own @-mention of
    the brand was silently triggering the "their app" keyword. See
    DECISION_LOG.md #7.
    """
    cleaned = strip_mentions_and_urls(text).lower()
    for intent, keywords in _RULES:
        for kw in keywords:
            if kw == "app":
                # "app" needs a word boundary: as a plain substring it also
                # matches inside "applesupport", so it used to fire on every
                # message just for mentioning the brand. Everything else
                # below is matched as a deliberate substring (e.g. "freez"
                # is meant to catch freeze/freezes/freezing).
                matched = re.search(r"\bapp\b", cleaned) is not None
            else:
                matched = kw in cleaned
            if matched:
                return intent, f"matched keyword rule for '{intent}' (keyword: '{kw}')"
    return "other", "no keyword rule matched"


def trivial_intent(_text: str) -> tuple[str, str]:
    """Baseline 1 (trivial): always predict the majority class."""
    return "update_performance", "majority-class baseline, ignores message content"
