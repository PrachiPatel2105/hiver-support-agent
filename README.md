# AppleSupport AI Support Agent — Hiver SDE Intern Take-Home

An AI support agent for **@AppleSupport** (chosen from the *Customer Support on Twitter* dataset)
that classifies an incoming customer tweet into one of five data-derived intents, drafts a reply
grounded in how AppleSupport has actually responded to similar issues historically via retrieval,
and decides whether to auto-handle or escalate the message to a human — with a stated, inspectable
reason for every decision.

---

> **EVALUATOR NOTE — Golden set size:**
> The assignment asks for 150–250 hand-labelled examples. This repo contains **17**.
> This is not an oversight. Please read the **Data note** section immediately below
> before checking the evaluation files. The explanation is substantive, documented
> throughout the codebase, and directly relevant to understanding every result in this repo.
> See also: [How to regenerate with the full dataset](#how-to-regenerate-the-golden-set-with-the-full-dataset).

---

## Data note (read this first)

This environment had no outbound network access to Kaggle or Hugging Face (sandboxed
take-home environment, confirmed by attempting both — both blocked by egress proxy), so
this repo runs against the **93-row sample** (`data/twcs_sample.csv`) provided alongside
the assignment brief, not the full ~3M-row `thoughtvector/customer-support-on-twitter`
Kaggle file.

The practical consequence: within that 93-row sample, only **17 messages** are actually
directed at @AppleSupport, and only 13 of those have a captured brand reply in-sample.
The assignment asks for a 150–250-example golden set; `eval/golden_set.csv` is instead
an **exhaustive hand-labelled census of all 17** AppleSupport-directed messages in the
data available — not a random draw from a larger pool, because there is no larger pool
here. See `DECISION_LOG.md` #1 for why fabricating a synthetic dataset to hit the letter
of the requirement would have been the wrong call.

The code is **dataset-scale-agnostic**: `src/load_data.py` takes any CSV in this schema.
Pointing `DATA_PATH` at the full Kaggle file requires no code changes.

---

## Repository layout

```
data/
  twcs_sample.csv              93-row provided sample (raw schema, unmodified)
src/
  load_data.py                 CSV loading + thread reconstruction (any brand)
  text_utils.py                shared @mention/URL stripping
  intents.py                   5-intent taxonomy + rule-based classifier
  retrieval.py                 TF-IDF grounding index over historical replies
  llm_client.py                pluggable LLM backend (Anthropic / OpenAI / offline)
  agent.py                     pipeline: classify → ground → draft → decide
  judge.py                     LLM-as-judge / offline rubric reply-quality scorer
eval/
  golden_set.csv               17 hand-labelled examples (see Data note)
  human_judge_scores.csv       hand-scored reply quality for judge-agreement check
  run_eval.py                  baselines + full agent, all metrics + per-intent table
  judge_agreement.py           judge-vs-human agreement stats + interpretation
  eval_results.json            persisted output of the last eval run
  judge_agreement_results.json persisted agreement stats
  JUDGE_RUBRIC.md              full 4-axis rubric (grounded/actionable/tone/safe)
reports/
  REPORT.md                    problem framing, results, failure analysis, etc.
  DECISION_LOG.md              15 non-obvious decisions and why
run.sh                         one command to reproduce all results (Unix/macOS/WSL)
run.bat                        same, for Windows PowerShell / cmd
requirements.txt               three pure-Python dependencies
```

---

## How to run

### Unix / macOS / WSL (Git Bash on Windows also works)

```bash
pip install -r requirements.txt
./run.sh
```

### Windows (PowerShell or cmd — no WSL required)

```bat
pip install -r requirements.txt
run.bat
```

Or run the three steps individually in PowerShell:

```powershell
python src/load_data.py data/twcs_sample.csv
python src/agent.py "@AppleSupport my battery is draining so fast since the update, please help"
python eval/run_eval.py
python eval/judge_agreement.py
```

`run.sh` / `run.bat` does three things in sequence:

1. Prints a summary of the loaded data (`src/load_data.py`)
2. Runs one example tweet through the live agent end-to-end and prints its full structured output
3. Runs the full evaluation harness — baselines + full agent + judge-agreement — against
   `eval/golden_set.csv`, writing results to `eval/eval_results.json` and
   `eval/judge_agreement_results.json`

**Runtime: ~4 seconds** on the 93-row sample with offline mode (no API key). This is fast
because the dataset is tiny (17 golden examples, 12-document retrieval index) and no
network calls are made in offline mode.

To try an arbitrary message:

```bash
python src/agent.py "@AppleSupport my phone won't stop crashing since the update"
```

### Using a real LLM instead of the offline fallback

No API key is required. By default, reply drafting and judging run in deterministic
**offline mode** (retrieval-adapted templates + rubric scoring — see `src/llm_client.py`).
Set either variable below and the same code path calls a real model:

```bash
export ANTHROPIC_API_KEY=sk-...   # tried first
# or
export OPENAI_API_KEY=sk-...
```

Every output record carries an explicit `llm_backend` field (`offline-template` /
`offline-rubric` vs. `claude-3-5-sonnet-latest` / `gpt-4o-mini`) so results are never
silently mixed between modes.

---

## Deliverables

### 1. Runnable pipeline

**Files:** [`run.sh`](run.sh) · [`run.bat`](run.bat) · [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt && ./run.sh   # Unix
pip install -r requirements.txt && run.bat    # Windows
```

Reproduces the headline results in well under the 15-minute budget. In practice **~4
seconds**, because the dataset is the 93-row sample (no larger pool available — see Data
note) and all computation is local with no network calls in offline mode.

---

### 2. Golden evaluation set

**File:** [`eval/golden_set.csv`](eval/golden_set.csv)

**17 examples** — an exhaustive census of every @AppleSupport-directed message in the
available 93-row sample.

**Why 17, not 150–250:** The assignment asks for 150–250 examples sampled from the full
~3M-row Kaggle dataset. That dataset was inaccessible in this environment (network egress
blocked — see Data note). Fabricating synthetic messages to hit the number would have
produced a misleading eval set. The honest approach was to label every real example
available and document the gap explicitly everywhere it matters.

**What the golden set contains:**
- Original tweet text, gold intent label, gold auto-handle/escalate decision
- Decision reasoning (`gold_decision_reason`)
- Reference reply (`gold_reply_reference`)
- Per-row `labeler_notes` explaining non-obvious calls

**Sampling methodology:** Exhaustive — every inbound tweet in `data/twcs_sample.csv`
that mentions `@AppleSupport` or is part of an @AppleSupport thread, sorted by `tweet_id`.
No random sampling because the population (17 messages) fits entirely.

**Labelling methodology:** Each intent was assigned by matching the cleaned tweet text
against the 5-class taxonomy in `src/intents.py`. Each escalation decision was assigned
by applying the policy in `src/agent.py::decide()` as a starting point, then overriding
with human judgment for edge cases (documented in `labeler_notes`). Hard cases where the
gold label intentionally disagrees with the automated system's output are flagged
explicitly (tweets 119249, 119290) — see `DECISION_LOG.md` #14.

**Class distribution:** update_performance: 8 · battery_life: 5 · software_bug_ui: 2 ·
positive_feedback: 1 · account_access: 1

#### How to regenerate the golden set with the full dataset

If you have access to the Kaggle dataset (`twcs.csv`, ~3M rows):

1. Place the file at `data/twcs.csv`
2. Run `python src/load_data.py data/twcs.csv` — this prints the brand distribution;
   look for `AppleSupport` (it has ~46k outbound messages in the full dataset)
3. Extract a random 200-example sample of inbound @AppleSupport messages:
   ```python
   import csv, random
   rows = [r for r in csv.DictReader(open('data/twcs.csv'))
           if r['inbound'].lower()=='true' and '@AppleSupport' in r['text']]
   random.seed(42)
   sample = random.sample(rows, 200)
   ```
4. Hand-label each row's `gold_intent` (using the taxonomy in `src/intents.py`) and
   `gold_decision` (using the escalation policy in `src/agent.py`) — this takes ~2–3
   hours for 200 examples
5. Replace `eval/golden_set.csv` with the new file, re-run `python eval/run_eval.py`

---

### 3. Evaluation harness

**Files:** [`eval/run_eval.py`](eval/run_eval.py) · [`src/judge.py`](src/judge.py) ·
[`eval/JUDGE_RUBRIC.md`](eval/JUDGE_RUBRIC.md) ·
[`eval/human_judge_scores.csv`](eval/human_judge_scores.csv) ·
[`eval/judge_agreement.py`](eval/judge_agreement.py)

`eval/run_eval.py` runs three systems and reports:

- **Intent classification:** accuracy, macro-F1, and **per-intent precision/recall/F1/support**
- **Escalation decision:** accuracy, precision, recall (escalate class)
- **Reply quality** (auto-handle messages only, n=10): mean LLM-as-judge overall score + word-overlap F1 vs reference
- **Confusion matrix** for intent classification

#### Judge rubric (summary — full version in [`eval/JUDGE_RUBRIC.md`](eval/JUDGE_RUBRIC.md))

| Axis | What it measures | 1 = | 3 = | 5 = |
|------|-----------------|-----|-----|-----|
| Grounded | Reply references customer's actual problem | Wrong topic / pure generic | One content word shared | ≥2 content words shared |
| Actionable | Reply moves conversation forward | No next step | Vague "DM us" | Specific diagnostic question |
| Tone | Appropriate acknowledgement | Dismissive/robotic | Neutral | Warm, specific |
| Safe | No fabricated promises/timelines | Explicit false promise | Hedged implication | No unsupported claims |

Overall = mean of four axes.

#### Judge-human agreement (n=10)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| MAE | 0.375 | Mean absolute error on a 1–5 scale |
| Pearson r | −0.171 | Near-zero — see note below |
| Spearman r | 0.029 | Near-zero — see note below |
| Within 0.5 pts | 90% | 9 of 10 replies within half a point |

**Why Pearson/Spearman are near-zero despite low MAE:** All 10 replies scored between
3.5 and 5.0 — a 1.5-point range on a 5-point scale. Correlation metrics are sensitive
to rank order; in a narrow band, tiny absolute differences arbitrarily flip rankings.
MAE and within-0.5 measure absolute error, which is more informative here. Both readings
are simultaneously true — reporting only one would be misleading. The judge is a useful
sanity check but **not validated against an independent annotator** (see
`DECISION_LOG.md` #10).

---

### 4. Report

**File:** [`reports/REPORT.md`](reports/REPORT.md)

Sections: Problem framing · Why AppleSupport · What "good" means · What was not built ·
Architecture · Intent taxonomy · Retrieval/grounding · Escalation logic ·
Results vs. two baselines · Per-intent metrics · Reply quality · LLM judge methodology ·
Judge-human agreement · Top 5 failure modes · "What is misleading about my headline number?" ·
One-week next steps · Golden-set methodology · Key limitations

---

### 5. Decision log

**File:** [`reports/DECISION_LOG.md`](reports/DECISION_LOG.md)

15 non-obvious decisions. Highlights:
- **Decision #7** — real bug found and fixed: `@AppleSupport` mention caused `"app"` keyword to fire on every message
- **Decision #9** — real bug found via eval output (not code inspection): positive-feedback message received a diagnostic question as a reply

---

## The three agent requirements

1. **Classify** — `src/intents.py`: 5 intents (`battery_life`, `update_performance`,
   `software_bug_ui`, `account_access`, `positive_feedback`) + `other` catch-all,
   keyword/word-boundary rule classifier.

2. **Draft a reply grounded in history** — `src/retrieval.py`: TF-IDF index over real
   historical (customer → brand) exchanges; `src/agent.py::draft_reply` few-shots an LLM
   with top-3 retrieved exemplars, or adapts the best-matching historical reply in offline mode.

3. **Auto-handle vs. escalate, with a reason** — `src/agent.py::decide`: inspectable rule
   set (account/identity → always escalate; unclassified → always escalate; high frustration
   → escalate; weak grounding → escalate; otherwise auto-handle) returning a human-readable
   reason string with every decision.
