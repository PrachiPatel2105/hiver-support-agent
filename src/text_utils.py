"""Shared tweet text cleanup, used by both the intent classifier and the
retrieval index so the two don't diverge in how they see the same text."""
from __future__ import annotations

import re

_MENTION_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"https?://\S+")


def strip_mentions_and_urls(text: str) -> str:
    """
    Remove @handles and URLs.

    This matters more than it looks: leaving "@AppleSupport" in the text
    means the substring "app" (a keyword meant to catch "my app crashed")
    also fires on every single message directed at the brand, since
    "applesupport" contains "app". An early version of this pipeline had
    exactly that bug -- it silently mis-classified messages purely because
    they were addressed to @AppleSupport. See DECISION_LOG.md #7.
    """
    text = _MENTION_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return text
