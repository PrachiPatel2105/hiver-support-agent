# Report: AppleSupport AI Support Agent

## Problem framing

**Brand:** @AppleSupport.

**Why AppleSupport:** It has the most outbound messages in the provided 93-row
sample (13 replies, vs. 8 for the next two brands), which maximises the amount
of real historical grounding data available when the whole dataset is this
small. See `DECISION_LOG.md` #2.

**What "good" means for this brand:** In a public Twitter support queue, the
first-touch reply has one job — get the customer into a private, resolvable
channel (DM) with the right diagnostic information, without saying anything
false, dismissive, or premature. It is *not* the reply's job to actually
resolve the technical issue in one public tweet (AppleSupport's own real
historical replies never do this — they all move to DM). So "good" here means:
correct intent, a reply that references something specific to what the customer
said, a next step that moves the conversation forward, and — just as important
— knowing when *not* to auto-reply at all.

**What I chose not to build:** multi-turn DM conversation handling (the three
requirements are all about the first-touch triage decision); a general Apple
support taxonomy (the 5-class taxonomy here covers exactly the data seen, not
Apple's real internal categories); multi-label intents (several messages carry
more than one, see `DECISION_LOG.md` #15); and a sarcasm detector beyond one
narrow, documented fix (`DECISION_LOG.md` #12).

**The data constraint that shapes everything below:** this sandboxed environment
could not reach Kaggle or Hugging Face, so this whole system was built and
evaluated against the 93-row sample provided with the assignment, in which only
17 messages are directed at @AppleSupport. See `README.md` "Data note" and
`eval/golden_set.csv` for the full accounting. Every result in this report
should be read as "does the approach work on the data available," not "here is
a production-grade accuracy number" — which the "misleading headline" section
makes explicit.

---

## Architecture

The pipeline has four sequential stages, each in its own `src/` module:

```
customer tweet
    │
    ▼
[1] classify intent        src/intents.py  rule_based_intent()
    │  returns (intent, reason)
    ▼
[2] retrieve grounding     src/retrieval.py  ExchangeIndex.search()
    │  TF-IDF over historical (customer→brand) pairs
    │  leave-one-out during eval (DECISION_LOG.md #8)
    ▼
[3] draft reply            src/agent.py  draft_reply()
    │  LLM few-shot with top-3 retrieved exemplars
    │  offline: adapt single best-matching historical reply
    ▼
[4] decide auto/escalate   src/agent.py  decide()
       5 inspectable rule paths, each returns a human-readable reason
```

**LLM backend** (`src/llm_client.py`): Anthropic → OpenAI → offline-template
fallback, in that order. No API key required to run. Every output record carries
an explicit `llm_backend` field so results are never silently mixed between modes.

**Judge** (`src/judge.py`): same pluggable pattern — LLM judge when key is set,
deterministic 4-axis rubric scorer otherwise. Rubric fully documented in
`eval/JUDGE_RUBRIC.md`.

---

## Intent taxonomy

Five classes, defined by reading every customer message in the 17-example golden
set rather than designed top-down:

| Intent | Definition | n in golden set |
|--------|-----------|----------------|
| `update_performance` | Slowness, freezing, wifi issues, app crashes attributed to an iOS update | 8 |
| `battery_life` | Battery draining faster than expected, usually after an update | 5 |
| `software_bug_ui` | Specific reproducible bug/UI glitch not framed as update side-effect | 2 |
| `positive_feedback` | Problem resolved / thanks / praise | 1 |
| `account_access` | Account, App Store, iCloud verification or sign-in problems | 1 |
| `other` | No keyword rule matched — always escalated | 0 in gold |

`other` always escalates by design: an unclassified message is exactly the case
where auto-handling is riskiest (`DECISION_LOG.md` #4).

---

## Retrieval and grounding

`src/retrieval.py` builds a TF-IDF index (unigrams + bigrams, stop-words removed)
over the cleaned customer-message text of every historical @AppleSupport exchange
in the dataset (`src/text_utils.strip_mentions_and_urls` applied first — see
`DECISION_LOG.md` #7 for why this matters). At query time, cosine similarity
ranks all historical exchanges and the top-3 are returned.

`draft_reply()` uses these as few-shot exemplars in the LLM prompt: the model
sees real historical `(customer said X → brand replied Y)` pairs and is asked
to produce a reply in the same voice for the new message. In offline mode the
single best-matching historical reply's text is adapted structurally.

Every `AgentOutput` records `grounded_on` (the historical tweet IDs used) and
`top_grounding_score` (the cosine similarity of the best match), making the
grounding fully traceable.

**Limitation at current scale:** the index holds 12 historical exchanges.
TF-IDF has no semantic understanding; shared generic tokens (e.g. "update",
"phone") can rank an irrelevant exchange highly. This is failure mode #3 below.

---

## Escalation logic

`src/agent.py::decide()` is a small, inspectable rule set — not a model — because
requirement #3 asks for a *stated reason* and a stated reason must be traceable
by a human reviewer (`DECISION_LOG.md` #6):

| Condition | Decision | Reason string |
|-----------|----------|--------------|
| `intent == positive_feedback` | auto_handle | "positive/resolved message — safe to auto-acknowledge" |
| `intent == account_access` | escalate | "intent involves account/identity verification" |
| `intent == other` | escalate | "unclassified traffic defaults to human review" |
| frustration_score ≥ 2 | escalate | "high frustration signal detected (score=N)" |
| top_grounding_score < 0.12 | escalate | "no sufficiently similar historical resolution found" |
| else | auto_handle | "established historical resolution pattern (score=X)" |

`frustration_score` counts profanity words, angry emoji, repeated exclamation
marks, and shouting caps. Threshold and word-list are committed constants, not
model weights, so any reviewer can inspect and override them.

The escalation policy is intentionally stricter than what AppleSupport actually
did in 2017 (it sent the same generic boilerplate to profane messages and
account-security requests alike). See `DECISION_LOG.md` #5.

---

## Results vs. two baselines

All numbers from `eval/run_eval.py` against `eval/golden_set.csv` (17 examples),
with leave-one-out retrieval so no grounding score benefits from a query matching
itself in the index.

**Trivial baseline:** always predicts `update_performance`; always `auto_handle`.  
**Simple baseline:** same keyword classifier as the full agent; escalates only for
`account_access` and `other` — no tone signal, no grounding threshold.  
**Full agent:** intent classifier + retrieval-grounded drafting +
frustration/grounding-aware escalation.

### Intent classification

| System | Accuracy | Macro-F1 |
|--------|----------|---------|
| trivial | 0.471 | 0.128 |
| simple | 0.941 | 0.822 |
| **agent** | **0.941** | **0.822** |

Intent accuracy and macro-F1 are **identical between simple and agent** because
both use the exact same `rule_based_intent()` classifier. The agent's entire
improvement over "simple" is in the escalation layer, not in classification.
A reviewer who reads only the intent row would incorrectly conclude the agent adds
nothing over a keyword baseline.

### Per-intent breakdown (agent / simple — same classifier)

| Intent | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| account_access | 1.000 | 1.000 | 1.000 | 1 |
| battery_life | 1.000 | 1.000 | 1.000 | 5 |
| other | 0.000 | 0.000 | 0.000 | 0 |
| positive_feedback | 1.000 | 1.000 | 1.000 | 1 |
| software_bug_ui | 1.000 | 1.000 | 1.000 | 2 |
| update_performance | 1.000 | 0.875 | 0.933 | 8 |

The single misclassification: tweet 119249 (`gold=update_performance`,
`pred=other`). This is failure mode #1 — the message has no keyword signal
without its parent tweet's context ("Me too am suffering, hope they can find a
solution"). The `other` row has 0 support in the golden set and 0 precision
because no message was labelled `other` by the human, though the system predicts
it for tweet 119249.

### Confusion matrix (agent)

```
pred →         account_access  battery_life  other  positive_feedback  software_bug_ui  update_performance
account_access      1               0          0          0                 0                 0
battery_life        0               5          0          0                 0                 0
other               0               0          0          0                 0                 0
positive_feedback   0               0          0          1                 0                 0
software_bug_ui     0               0          0          0                 2                 0
update_performance  0               0          1          0                 0                 7
```

One off-diagonal: update_performance → other (tweet 119249).

### Escalation decision

| System | Accuracy | Escalate precision | Escalate recall |
|--------|---------|--------------------|----------------|
| trivial | 0.588 | 0.000 | 0.000 |
| simple | 0.706 | 1.000 | 0.286 |
| **agent** | **0.941** | **1.000** | **0.857** |

The real separation between simple and agent is escalation recall: 0.286 → 0.857.
The simple baseline's blind spot is every high-frustration message (119253,
119268, 119272) — it has no tone signal and sends those the same boilerplate as a
calm complaint.

### Reply quality (10 auto-handle messages)

| Metric | Value |
|--------|-------|
| Mean judge overall | 4.0 / 5 |
| Mean word-overlap F1 vs reference | 0.409 |
| n scored | 10 |

Reply quality is judged only for gold-labelled `auto_handle` messages (the only
case where the drafted reply would actually be sent). See `eval/JUDGE_RUBRIC.md`
for the full 4-axis rubric.

---

## LLM judge methodology

`src/judge.py` scores each reply on four axes (grounded, actionable, tone, safe),
each 1–5, overall = mean. Full definitions and examples in `eval/JUDGE_RUBRIC.md`.

When no API key is set, `_rubric_judge()` implements the same four axes
deterministically (lexical checks, presence of `?`/`DM`, acknowledgement
phrases, over-promise keywords). All 17 eval rows were scored with the offline
rubric.

The LLM judge prompt (in `_llm_judge()`) injects the same four-axis descriptions
and asks for compact JSON output. On unparseable output it falls back to the
rubric scorer and labels the backend accordingly.

---

## Human-vs-judge agreement

`eval/judge_agreement.py` computes agreement between the offline rubric judge and
hand-scores for the same 10 auto-handled replies. Scores were written in
`eval/human_judge_scores.csv` *before* looking at the judge's numbers.

| Metric | Value |
|--------|-------|
| n | 10 |
| MAE | 0.375 |
| Pearson r | −0.171 |
| Spearman r | 0.029 |
| Within 0.5 pts | 90% |

**Why Pearson/Spearman are near-zero despite low MAE:** All 10 replies scored
between 3.5 and 5.0 (a 1.5-point spread on a 5-point scale). Correlation metrics
are sensitive to rank order; in a narrow band, tiny absolute differences
arbitrarily flip rankings. MAE and within-0.5 measure absolute error, which is
more informative at this scale. Both readings are simultaneously true.

**What this agreement means and doesn't mean:** The "human" here is the same
person who built the judge and the golden set. This is a self-consistency check
that caught two real failures (tweets 119291 and 119326), but it is not
inter-annotator agreement with an independent reviewer. See `DECISION_LOG.md` #10.

---

## Failure analysis: top 5

### 1. No thread/conversation context (tweet 119249)

**Text:** "Me too am suffering, hope they can find a solution"  
**Expected:** classify as `update_performance`, note thread dependency, escalate  
**Actual:** classified as `other` (no keyword match), escalated for the wrong reason  
**Why:** The message is a reply in the same thread as tweet 119250's iOS-slowness
complaint. Read in isolation it has no symptom signal. The classifier correctly
has no idea what to do with it — the decision (escalate) happens to be right but
the reason is wrong ("unclassified" rather than "needs parent tweet context").  
**Hypothesis:** Single-tweet classification with no conversation context is
structurally incapable of handling thread replies. Fix: pass parent-tweet text
as additional context to the classifier.

### 2. Customer already tried the documented fix (tweet 119290)

**Text:** "I have read the help page, turned off virtually all apps and location
services too — but in typing this I dropped 3% life"  
**Expected:** escalate — auto-sending standard troubleshooting is redundant  
**Actual:** auto_handle (frustration score 1, grounding score 0.37 — both below
escalation thresholds)  
**Why:** The rule set has no signal for "customer explicitly states they already
followed documented steps." High grounding score is actively misleading here —
the message shares vocabulary with other battery complaints, so it retrieves a
"which iOS version are you on?" reply, which the customer has effectively already
answered.  
**Hypothesis:** Requires a short NLU pass over the message to detect
"already tried X" phrasing before falling through to the standard draft.

### 3. Retrieval grounds on wrong exemplar (tweet 119326)

**Text:** "My apps stop working without warning and my phone freezes every five
minutes! Love the new update!!!!"  
**Drafted reply:** "What happens when you try to listen to Apple Music & use
WhatsApp? Does the music pause?"  
**Expected:** ask about freezing/crashing  
**Why:** TF-IDF over 12 documents matched shared generic tokens (not the
freezing/crashing symptom) against a Music+WhatsApp exchange. The human judge
gave grounding a 1/5 — "actively grounded on the wrong exchange."  
**Hypothesis:** Vocabulary overlap ≠ semantic relevance at 12-document corpus
scale. A small sentence-embedding bi-encoder would fix this.

### 4. Single-path drafting for unseen intents (tweet 119291, found via eval)

**Text:** "Super help — problem solved, once again in love with Apple"  
**Before fix:** drafted a diagnostic troubleshooting question ("When did this
start happening?") to a customer who just said their problem is solved.  
**Why:** The retrieval corpus contains zero "thanks/resolved" exchanges — every
historical reply is a troubleshooting response to an open complaint. The
retrieval-grounded drafter dutifully adapted the best-matching exchange.  
**Fix applied:** `positive_feedback` is now special-cased in `run_agent()` to
bypass retrieval entirely and return a fixed acknowledgement template.  
**General lesson:** Any intent under-represented (or absent) in the historical
grounding corpus is a drafting landmine. See `DECISION_LOG.md` #9.

### 5. Escalation policy stricter than historical brand behaviour (tweets 119253, 119268, 119272, 119299)

**Actual 2017 AppleSupport behaviour:** Sent the identical generic
"thanks for reaching out, DM us" boilerplate to profane messages and to an
account-security request alike — the same response regardless of tone or
security risk.  
**Agent behaviour:** Escalates profanity/high-frustration messages and always
escalates account/identity requests.  
**Why this is a design decision, not a failure:** If "match the brand's real
historical behaviour" were the metric, these four would count as misses. The
golden labels were set against a deliberate judgment that customer-facing tone
risk and account-security risk are real product requirements worth being stricter
about, even if the brand's 2017 social-media playbook didn't optimise for them.
See `DECISION_LOG.md` #5.

---

## What is misleading about my headline number

Three separate things, not one:

**"94.1% escalation accuracy" is inflated by construction.** 17 examples,
hand-labelled by the same person who wrote the escalation thresholds, while
looking at the same 17 messages. The thresholds (frustration word list, 2-point
cutoff, 0.12 grounding cutoff) were tuned on exactly this eval set. That is the
textbook setup for a rule set to look better on its own eval than on genuinely
new data. Two small countermeasures were applied — leave-one-out retrieval
(`DECISION_LOG.md` #8) and deliberately seeding known disagreements into the
golden set (`DECISION_LOG.md` #14) — but neither makes 94.1% trustworthy the
way a held-out result on hundreds of independently-labelled examples would be.

**Intent accuracy being identical between simple and agent (0.941) can look like
"the agent adds nothing over a keyword rule."** It doesn't — the full
improvement is in escalation recall (0.286 → 0.857). A reader who skims only
the intent-accuracy row leaves with the wrong takeaway.

**The judge-agreement statistics look almost contradictory, and that's
informative.** Pearson r = −0.171 (no rank correlation) but MAE = 0.375 and 90%
within 0.5 points. Both are true simultaneously because all 10 replies scored in
a 1.5-point band (3.5–5.0 on a 1–5 scale). Correlation is sensitive to rank
order; MAE is sensitive to absolute error. Reporting only the correlation makes
the judge look useless; reporting only the within-0.5 makes it look validated.
The honest reading: the offline judge correctly flagged the two biggest
reply-quality failures in this dataset (tweets 119291 and 119326) but has not
been validated against an independent human annotator.

---

## Golden-set methodology

**Sampling:** Exhaustive census — every inbound tweet in `data/twcs_sample.csv`
that mentions `@AppleSupport` or is part of an @AppleSupport thread, sorted by
`tweet_id`. No random sampling because the entire population (17 messages) fits.
This is not a random sample of a larger pool because the larger pool
(the full ~3M-row Kaggle dataset) was inaccessible in this environment.

**Labelling:** Intent assigned by matching stripped tweet text against the 5-class
taxonomy. Escalation decision assigned by applying the `decide()` rules as a
starting point, then applying human judgment to override edge cases. Each
non-obvious label is explained in the `labeler_notes` column. Two "hard case"
examples where the gold label intentionally disagrees with the automated
system's output are flagged explicitly (tweets 119249 and 119290) to avoid the
eval set grading its own homework.

**Quality control:** Leave-one-out retrieval so no grounding score benefits from
a query retrieving itself. All 17 labeller notes reviewed for internal
consistency. Two deliberate disagreement seeds as described above.

**Key limitation:** Single annotator. No inter-annotator agreement measurement.
All labels, thresholds, and golden set produced by the same person. Treat all
metrics as self-reported, not independently validated.

---

## Key limitations

1. **17-example golden set** (assignment asks for 150–250). Root cause:
   sandboxed environment, no Kaggle access. See `DECISION_LOG.md` #1 and
   `README.md` for regeneration instructions once the full dataset is available.

2. **Single annotator** for golden set, human judge scores, and escalation
   thresholds. No independent inter-rater agreement.

3. **TF-IDF retrieval over 12 documents** is too small for semantic precision.
   Generic token overlap can retrieve the wrong historical exchange (failure
   mode #3).

4. **Rule-based intent classifier** derived from 17 examples will miss novel
   phrasing. No learned generalisation beyond keyword matching.

5. **Offline judge** cannot distinguish "no topical overlap" (generic reply,
   bad) from "wrong-topic overlap" (topically wrong reply, worse). See
   `eval/JUDGE_RUBRIC.md` known weaknesses.

---

## What I'd do with one more week

In rough priority order:

1. **Get the full Kaggle dataset** (unrestricted egress or local download),
   re-derive the intent taxonomy against a much larger AppleSupport slice, and
   build a genuine 150–250-example random golden set with sampling documented
   and a held-out subset never seen during threshold tuning.
2. **Add a second independent annotator** for both the golden set and the
   judge-agreement subset. Right now both are the same person.
3. **Replace TF-IDF retrieval with a small sentence-embedding bi-encoder** to
   fix failure mode #3 (grounding on a topically wrong exemplar). Specifically
   targets the Music+WhatsApp retrieval miss on tweet 119326.
4. **Add a "customer already tried standard troubleshooting" detector** for
   failure mode #2. Likely a short LLM classification pass over the message
   rather than a keyword rule, since the phrasing varies widely.
5. **Support multi-label intents** (`DECISION_LOG.md` #15) so compound messages
   like tweet 119270 (battery + crashes + downgrade request) don't force a
   single lossy label.
6. **Build a regression test suite** that pins the two bugs found during this
   build (the `@mention`/`"app"` substring collision and the
   positive-feedback drafting bug) so they cannot silently reappear.
