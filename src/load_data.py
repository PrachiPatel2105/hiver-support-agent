"""
Data loading + thread reconstruction for the Customer Support on Twitter dataset
(Kaggle: thoughtvector/customer-support-on-twitter).

Schema: tweet_id, author_id, inbound, created_at, text,
        response_tweet_id, in_response_to_tweet_id

This module is dataset-scale-agnostic: point it at the 93-row sample shipped in
data/twcs_sample.csv, or at the full ~3M-row Kaggle CSV, and it produces the same
structures. See README.md "Data note" for why this repo ships and runs against the
sample rather than the full file.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Tweet:
    tweet_id: str
    author_id: str
    inbound: bool
    created_at: str
    text: str
    response_tweet_ids: list[str] = field(default_factory=list)
    in_response_to_tweet_id: Optional[str] = None


@dataclass
class Exchange:
    """One (customer message -> brand reply) pair, i.e. a single support turn."""
    customer_tweet: Tweet
    brand_tweet: Tweet
    brand: str


def load_tweets(csv_path: str) -> dict[str, Tweet]:
    tweets: dict[str, Tweet] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["tweet_id"]
            tweets[rid] = Tweet(
                tweet_id=rid,
                author_id=row["author_id"],
                inbound=row["inbound"].strip().lower() == "true",
                created_at=row["created_at"],
                text=row["text"],
                response_tweet_ids=[t for t in row["response_tweet_id"].split(",") if t],
                in_response_to_tweet_id=row["in_response_to_tweet_id"] or None,
            )
    return tweets


def brand_authors(tweets: dict[str, Tweet]) -> list[str]:
    """Non-inbound (outbound) author_ids are brand handles."""
    seen = []
    for t in tweets.values():
        if not t.inbound and t.author_id not in seen:
            seen.append(t.author_id)
    return seen


def exchanges_for_brand(tweets: dict[str, Tweet], brand: str) -> list[Exchange]:
    """
    Reconstruct direct customer->brand exchanges for one brand: every brand tweet
    that is *in response to* an inbound customer tweet becomes one Exchange.
    This captures the first-touch turn, which is what the assignment asks the
    agent to handle (classify the incoming message, draft the reply, decide
    auto-handle vs escalate) -- not the full multi-turn DM thread that follows.
    """
    out = []
    for t in tweets.values():
        if t.inbound or t.author_id != brand:
            continue
        if not t.in_response_to_tweet_id:
            continue
        customer = tweets.get(t.in_response_to_tweet_id)
        if customer is None or not customer.inbound:
            continue
        out.append(Exchange(customer_tweet=customer, brand_tweet=t, brand=brand))
    return out


def inbound_messages_for_brand(tweets: dict[str, Tweet], brand: str) -> list[Tweet]:
    """
    All inbound customer tweets that are part of a thread involving `brand`,
    whether or not a brand reply exists in this slice of the data. This is the
    superset the *incoming* side of the agent has to handle -- includes cases
    where the brand never replied (in the sample) so we can still see what
    "no reply captured" looks like.
    """
    brand_ids = {t.tweet_id for t in tweets.values() if t.author_id == brand}
    out = []
    for t in tweets.values():
        if not t.inbound:
            continue
        mentions_brand = f"@{brand}" in t.text
        replies_touch_brand = any(rid in brand_ids for rid in t.response_tweet_ids)
        in_response_touches_brand = t.in_response_to_tweet_id in brand_ids
        if mentions_brand or replies_touch_brand or in_response_touches_brand:
            out.append(t)
    # de-dup, stable order
    seen_ids = set()
    deduped = []
    for t in out:
        if t.tweet_id not in seen_ids:
            deduped.append(t)
            seen_ids.add(t.tweet_id)
    return deduped


if __name__ == "__main__":
    import sys
    from collections import Counter

    path = sys.argv[1] if len(sys.argv) > 1 else "data/twcs_sample.csv"
    tweets = load_tweets(path)
    print(f"Loaded {len(tweets)} tweets from {path}")
    brands = Counter(t.author_id for t in tweets.values() if not t.inbound)
    print("Outbound (brand) author_id counts:")
    for b, n in brands.most_common():
        print(f"  {b}: {n}")
