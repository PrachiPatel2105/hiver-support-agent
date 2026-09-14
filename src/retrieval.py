"""
Retrieval over historical (customer message -> brand reply) exchanges, used to
ground drafted replies in how the brand has actually resolved similar issues
before (assignment requirement #2), rather than letting an LLM free-hand a
generic reply.

Approach: TF-IDF + cosine similarity over customer message text. This is a
handful of lines with scikit-learn, has zero external dependencies at
inference time, and -- importantly -- is fully inspectable: for every draft
we can point at exactly which historical tweet_id(s) it was grounded in,
which the "prove it works" framing of this assignment cares about a lot more
than a marginal accuracy gain from a fancier embedding model would.

At full dataset scale (see README "Data note") this index would hold
thousands of historical exchanges per brand and TF-IDF would likely be
swapped for a sentence-embedding model (e.g. a small bi-encoder) for better
recall on paraphrases -- noted in DECISION_LOG.md and REPORT.md "what I'd do
next".
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from load_data import Exchange
from text_utils import strip_mentions_and_urls as clean_for_retrieval


@dataclass
class RetrievalHit:
    exchange: Exchange
    score: float


class ExchangeIndex:
    def __init__(self, exchanges: list[Exchange]):
        self.exchanges = exchanges
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = [clean_for_retrieval(e.customer_tweet.text) for e in exchanges]
        self._matrix = self._vectorizer.fit_transform(corpus) if corpus else None

    def search(self, query_text: str, k: int = 3) -> list[RetrievalHit]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        q_vec = self._vectorizer.transform([clean_for_retrieval(query_text)])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        hits = [RetrievalHit(self.exchanges[i], float(sims[i])) for i in ranked[:k]]
        return hits
