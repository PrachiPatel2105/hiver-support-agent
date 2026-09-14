# AppleSupport AI Support Agent — Hiver SDE Intern Take-Home

An AI support agent for **@AppleSupport** (chosen from the *Customer Support on Twitter* dataset)
that classifies an incoming customer tweet into one of five data-derived intents, drafts a reply
grounded in how AppleSupport has actually responded to similar issues historically via retrieval,
and decides whether to auto-handle or escalate the message to a human — with a stated, inspectable
reason for every decision.

---

## Data note (read this first)

This environment had no outbound network access to Kaggle or Hugging Face (sandboxed take-home
environment, not a deliberate choice), so this repo runs against the **93-row sample**
(`data/twcs_sample.csv`) provided alongside the assignment brief, not the full ~3M-row
`thoughtvector/customer-support-on-twitter` Kaggle file. The assignment explicitly says a
subsample is expected ("We will not run your code on the full dataset"), so the code is written
to be dataset-scale-agnostic — `src/load_data.py` accepts any CSV in this exact schema, so
pointing `DATA_PATH` at the real Kaggle file (once downloaded) requires no code changes, only more
compute and time.

The practical consequence: within that 93-row sample, only **17 messages** are actually directed
at @AppleSupport, and only 13 of those have a captured brand reply in-sample. The assignment asks
for a 150–250-example golden set; `eval/golden_set.csv` is instead an **exhaustive hand-labelled
census of all 17** AppleSupport-directed messages available to this repo — not a random draw from a
larger pool, because there is no larger pool here. See `eval/golden_set.csv`'s notes column,
`reports/REPORT.md` ("What is misleading about my headline number"), and
`reports/DECISION_LOG.md` #1 for exactly what that does and does not tell you.

---

## Repository layout

```
data/
  twcs_sample.csv          93-row provided sample (raw schema, unmodified)
src/
  load_data.py             CSV loading + thread reconstruction (any brand)
  text_utils.py            shared @mention/URL stripping
  intents.py               5-intent taxonomy + rule-based classifier
  retrieval.py             TF-IDF grounding index over historical replies
  llm_client.py            pluggable LLM backend (Anthropic / OpenAI / offline)
  agent.py                 pipeline: classify → ground → draft → decide
  judge.py                 LLM-as-judge / offline rubric reply-quality scorer
eval/
  golden_set.csv           17 hand-labelled examples (see Data note above)
  human_judge_scores.csv   hand-scored reply quality for judge-agreement check
  run_eval.py              baselines + full agent, all metrics
  judge_agreement.py       judge-vs-human agreement stats
  eval_results.json        persisted output of the last eval run
  judge_agreement_results.json  persisted agreement stats
reports/
  REPORT.md                problem framing, results, failure analysis, etc.
  DECISION_LOG.md          15 non-obvious decisions and why
run.sh                     one command to reproduce all results
requirements.txt           three pure-Python dependencies
```

---

## How to run

```bash
pip install -r requirements.txt
./run.sh
```

`run.sh` does three things in sequence:

1. Prints a summary of the loaded data (`src/load_data.py`)
2. Runs one example tweet through the live agent end-to-end and prints its full structured output
3. Runs the full evaluation harness — baselines + full agent + judge-agreement — against
   `eval/golden_set.csv`, writing results to `eval/eval_results.json` and
   `eval/judge_agreement_results.json`

To try an arbitrary message directly:

```bash
python3 src/agent.py "@AppleSupport my phone won't stop crashing since the update"
```

### Using a real LLM instead of the offline fallback

No API key is configured by default, so reply drafting and judging run in deterministic
**offline mode** (retrieval-adapted templates + rubric scoring — see `src/llm_client.py`).
Set either variable below and the same code path calls a real model instead, with no flags:

```bash
export ANTHROPIC_API_KEY=sk-...   # tried first
# or
export OPENAI_API_KEY=sk-...
```

Every output record carries an explicit `llm_backend` field (`offline-template` /
`offline-rubric` vs. `claude-3-5-sonnet-latest` / `gpt-4o-mini`) so results are never silently
mixed between modes.

---

## Deliverables

### 1. Runnable pipeline

**Files:** [`run.sh`](run.sh) · [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt && ./run.sh
```

Reproduces the headline results in well under the 15-minute budget. In practice it completes in
**~4 seconds**, for two concrete reasons:

- The dataset is the 93-row sample (see Data note), not the multi-million-row Kaggle file —
  there is very little data to process.
- No network calls are made in offline mode. All three dependencies (`scikit-learn`, `numpy`,
  `scipy`) are pure-Python packages that install quickly and run entirely locally.

The pipeline itself has three named stages (classify → ground → draft → decide), each in its own
`src/` module, so the architecture is legible even at this scale.

---

### 2. Golden evaluation set

**File:** [`eval/golden_set.csv`](eval/golden_set.csv)

17 hand-labelled examples — an **exhaustive census** of every @AppleSupport-directed message in
the available 93-row sample, not a random draw from a larger pool (because no larger pool was
accessible — see the Data note above for why, and `DECISION_LOG.md` #1 for what changes with the
real dataset).

Each row carries: the original tweet text, a gold intent label, a gold auto-handle/escalate
decision, a reference reply, and a `labeler_notes` field that records the reasoning behind
non-obvious calls.

The set deliberately includes **"hard case" disagreements** — examples where the gold label
intentionally differs from what the automated system outputs (`DECISION_LOG.md` #14). An eval set
built entirely of cases the system is expected to get right grades its own homework; seeding known
disagreements (tweets 119249 and 119290) is a small explicit countermeasure against that.

---

### 3. Evaluation harness

**Files:** [`eval/run_eval.py`](eval/run_eval.py) · [`src/judge.py`](src/judge.py) ·
[`eval/human_judge_scores.csv`](eval/human_judge_scores.csv) ·
[`eval/judge_agreement.py`](eval/judge_agreement.py)

`eval/run_eval.py` runs three systems against the golden set and reports:

- **Intent classification:** accuracy and macro-F1
- **Escalation decision:** accuracy, precision, and recall on the escalate class
- **Reply quality** (auto-handle messages only, n = 10): mean LLM-as-judge overall score
  (offline rubric fallback when no API key is set) and mean word-overlap F1 against a reference
  reply

Results from `eval/eval_results.json` (the last persisted run):

| System  | Intent acc. | Intent macro-F1 | Escalation acc. | Escalate precision | Escalate recall |
|---------|-------------|-----------------|-----------------|-------------------|-----------------|
| trivial | 0.471       | 0.128           | 0.588           | 0.000             | 0.000           |
| simple  | 0.941       | 0.822           | 0.706           | 1.000             | 0.286           |
| **agent** | **0.941** | **0.822**       | **0.941**       | **1.000**         | **0.857**       |

Reply quality (10 auto-handle messages): mean judge score **4.0 / 5**, mean word-overlap F1
**0.409**.

`eval/judge_agreement.py` computes judge-vs-human agreement explicitly and reports it — it is not
asserted, it is measured. The agreement stats (Pearson r, Spearman r, MAE, % within 0.5 pts) are
written to `eval/judge_agreement_results.json`. See `reports/REPORT.md` ("What is misleading about
my headline number") for an honest interpretation of what those numbers mean and don't mean, given
that both the judge and the human scores here come from the same person.

---

### 4. Report

**File:** [`reports/REPORT.md`](reports/REPORT.md)

Contains the following sections, in order:

1. **Problem framing** — what "good" means for @AppleSupport on Twitter, what was explicitly
   scoped out and why, and the data constraint that shapes every result in the report
2. **Results vs. two baselines** — full metrics table (trivial baseline, simple keyword baseline,
   full agent) with an honest reading of where the agent actually improves over the simplest
   baseline and where it doesn't
3. **Failure analysis** — top 5 failure modes with specific tweet IDs, root causes, and
   cross-references to `DECISION_LOG.md` where relevant
4. **What is misleading about my headline number** — mandatory section; three separate things,
   not one: why 94.1% escalation accuracy is inflated by eval-set construction, why the
   judge-agreement statistics look almost contradictory (and why both readings are simultaneously
   true), and why "judge agrees with a human" here means "agrees with the person who built the
   whole pipeline"
5. **What I'd do with one more week** — six concrete priorities in rough order, starting with
   getting the real Kaggle dataset

---

### 5. Decision log

**File:** [`reports/DECISION_LOG.md`](reports/DECISION_LOG.md)

Documents **15 non-obvious decisions** made during development, numbered for cross-referencing
from code comments and `REPORT.md`. Includes, among others:

- Why @AppleSupport was chosen as the brand (#2)
- Why LLM calls degrade gracefully with no API key (#3)
- Why the escalation policy is intentionally stricter than what the brand historically did (#5)
- Why the escalation logic is a small rule set rather than a learned classifier (#6)
- **Two real bugs found and fixed during development:**
  - **Bug #7** — `@AppleSupport` mentions polluted both retrieval similarity scores and the
    `"app"` intent keyword (which matched as a substring of `"applesupport"` on literally every
    message); fixed by stripping `@handles`/URLs before indexing and adding a word-boundary match
    for the `"app"` keyword in `src/intents.py` and `src/text_utils.py`
  - **Bug #9** — positive-feedback messages ("problem solved!") were getting a diagnostic
    troubleshooting question as a reply, because the retrieval corpus contains zero
    resolved/positive exchanges to ground against; surfaced by `eval/run_eval.py` output (not by
    reading the code), fixed by special-casing `positive_feedback` in `src/agent.py` to skip
    retrieval-grounded drafting entirely
- Why the golden set deliberately includes hard-case disagreements (#14)
- Why multi-label intents were scoped out (#15)

---

## The three agent requirements, and where they live

1. **Classify** — `src/intents.py` defines 5 intents read off the actual data
   (`battery_life`, `update_performance`, `software_bug_ui`, `account_access`,
   `positive_feedback`, plus an `other` catch-all), with a keyword/word-boundary rule classifier.

2. **Draft a reply grounded in history** — `src/retrieval.py` indexes every real historical
   (customer message → AppleSupport reply) pair from the sample with TF-IDF; `src/agent.py::draft_reply`
   either few-shots an LLM with the top-3 retrieved exemplars or (offline mode) adapts the
   single best-matching historical reply's structure.

3. **Auto-handle vs. escalate, with a reason** — `src/agent.py::decide` is a small, inspectable
   rule set (identity/account issues always escalate; unclassified messages always escalate;
   high-frustration language escalates; weak retrieval grounding escalates; everything else
   auto-handles) that returns a human-readable reason string alongside every decision.
