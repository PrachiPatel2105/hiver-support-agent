# AppleSupport AI Support Agent -- Hiver SDE Intern Take-Home

An AI support agent for **@AppleSupport** (chosen from the *Customer Support
on Twitter* dataset) that classifies an incoming customer tweet into one of
five data-derived intents, drafts a reply grounded in how AppleSupport has
actually resolved similar issues before, and decides whether to auto-handle
or escalate the message to a human, with a stated reason for every decision.

## Data note (read this first)

This environment had no outbound network access to Kaggle or Hugging Face
(sandboxed take-home environment, not a deliberate choice), so this repo
runs against the **93-row sample** (`data/twcs_sample.csv`) that was
provided alongside the assignment brief, not the full ~3M-row
`thoughtvector/customer-support-on-twitter` Kaggle file. The assignment
explicitly says a subsample is expected ("We will not run your code on the
full dataset"), so the code is written to be dataset-scale-agnostic --
`src/load_data.py` takes any CSV in this exact schema as input, so pointing
`DATA_PATH` at the real Kaggle file (once downloaded) requires no code
changes, only more compute/time.

The practical consequence: within that 93-row sample, only **17 messages**
are actually directed at AppleSupport, and only 13 of those have a captured
brand reply in-sample. The assignment asks for a 150-250-example golden set;
`eval/golden_set.csv` is instead an **exhaustive hand-labelled census of all
17** AppleSupport-directed messages in the data available to this repo, not
a random sample of a larger pool. See `eval/golden_set.csv`'s own notes
column and `reports/REPORT.md` ("what's misleading about my headline
number") for exactly what that does and doesn't tell you, and
`DECISION_LOG.md` #1 for how this would change with the real dataset.

## What's in here

```
data/twcs_sample.csv     the provided 93-row sample (raw schema, unmodified)
src/
  load_data.py            CSV loading + thread reconstruction (any brand)
  text_utils.py           shared @mention/URL stripping
  intents.py              the 5-intent taxonomy + rule-based classifier
  retrieval.py            TF-IDF grounding index over historical replies
  llm_client.py           pluggable LLM backend (Anthropic/OpenAI/offline)
  agent.py                ties it together: classify -> ground -> draft -> decide
  judge.py                LLM-as-judge / offline rubric reply-quality scorer
eval/
  golden_set.csv           17 hand-labelled examples (see Data note above)
  human_judge_scores.csv   hand-scored reply quality, for judge agreement
  run_eval.py              baselines + full agent, all metrics
  judge_agreement.py       judge-vs-human agreement stats
reports/
  REPORT.md                problem framing, results, failure analysis, etc.
  DECISION_LOG.md           10-15 non-obvious decisions and why
run.sh                     reproduces everything below in one command
```

## Running it (< 15 minutes -- in practice, a few seconds)

```bash
pip install -r requirements.txt
./run.sh
```

This (1) prints a summary of the loaded data, (2) runs one example tweet
through the live agent end-to-end and prints its full structured output, and
(3) runs the evaluation harness (baselines + full agent + judge agreement)
against `eval/golden_set.csv` and writes `eval/eval_results.json` and
`eval/judge_agreement_results.json`.

To try your own message:

```bash
python3 src/agent.py "@AppleSupport my phone won't stop crashing since the update"
```

### Using a real LLM instead of the offline fallback

By default no LLM API key is configured in this environment, so reply
drafting and judging run in a deterministic **offline mode** (see
`src/llm_client.py`) that adapts real historical replies via retrieval
instead of free-generating text. Set either of these and the same code path
calls a real model instead, no flags needed:

```bash
export ANTHROPIC_API_KEY=sk-...   # tried first
# or
export OPENAI_API_KEY=sk-...
```

Every output record carries an explicit `llm_backend` field
(`offline-template` / `offline-rubric` vs. `claude-3-5-sonnet-latest` /
`gpt-4o-mini`) so results are never silently mixed between the two modes.

## The three requirements, and where they live

1. **Classify** -- `src/intents.py` defines 5 intents read off the actual
   data (`battery_life`, `update_performance`, `software_bug_ui`,
   `account_access`, `positive_feedback`, plus an `other` catch-all), with a
   keyword/word-boundary rule classifier.
2. **Draft a reply grounded in history** -- `src/retrieval.py` indexes every
   real historical (customer message -> AppleSupport reply) pair from the
   sample with TF-IDF, and `src/agent.py::draft_reply` either few-shots an
   LLM with the top-3 retrieved exemplars or (offline mode) adapts the
   single best-matching historical reply's structure.
3. **Auto-handle vs. escalate, with a reason** -- `src/agent.py::decide` is a
   small, inspectable rule set (identity/account issues always escalate;
   unclassified messages always escalate; high-frustration language
   escalates; weak retrieval grounding escalates; everything else
   auto-handles) that returns a human-readable reason string alongside every
   decision.

See `reports/REPORT.md` for results, baselines, and failure analysis, and
`reports/DECISION_LOG.md` for the non-obvious calls made along the way.
