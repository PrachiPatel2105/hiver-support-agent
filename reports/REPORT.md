# Report: AppleSupport AI support agent

## Problem framing

**Brand:** @AppleSupport. **What "good" means for this brand:** in a public
Twitter support queue, the first-touch reply has one job -- get the customer
into a private, resolvable channel (DM) with the right diagnostic
information, without saying anything false, dismissive, or premature. It is
*not* the reply's job to actually resolve the technical issue (AppleSupport's
own real historical replies never do this in one public tweet either -- they
all move to DM). So "good" here means: correct intent, a reply that
references something specific to what the customer said, a next step that
moves the conversation forward, and -- just as important -- knowing when
*not* to auto-reply at all.

**What I chose not to build:** multi-turn DM conversation handling (the
assignment's three requirements are all about the *first-touch* triage
decision); a general Apple support taxonomy (the taxonomy here is 5 classes
that cover exactly the data seen, not Apple's real internal categories);
multi-label intents (several messages carry more than one, see Decision Log
#15); and sarcasm/tone detection beyond one narrow, documented fix (Decision
Log #12).

**The data constraint that shapes everything below:** this sandboxed
environment could not reach Kaggle or Hugging Face, so this whole system was
built and evaluated against the 93-row sample provided with the assignment,
in which only 17 messages are actually directed at AppleSupport. See
`README.md` "Data note" and `eval/golden_set.csv` for the full accounting.
Every result in this report should be read as "does the approach work, on
the data available," not "here is a production-grade accuracy number,"
which the next section makes explicit.

## Results vs. two baselines

All numbers from `eval/run_eval.py` against `eval/golden_set.csv` (17
examples), with leave-one-out retrieval so no score benefits from a query
matching itself in the grounding corpus (Decision Log #8).

**Trivial baseline:** always predicts the majority intent
(`update_performance`) and always auto-handles.
**Simple baseline:** the same keyword intent classifier as the full agent,
but decides auto-handle/escalate purely from intent (`account_access`/`other`
escalate, everything else auto-handles -- no tone or grounding signal at
all).
**Full agent:** intent classifier + retrieval-grounded drafting +
frustration/grounding-aware escalation (`src/agent.py`).

| System  | Intent acc. | Intent macro-F1 | Escalation acc. | Escalate precision | Escalate recall |
|---|---|---|---|---|---|
| trivial | 0.471 | 0.128 | 0.588 | 0.000 | 0.000 |
| simple  | 0.941 | 0.822 | 0.706 | 1.000 | 0.286 |
| agent   | 0.941 | 0.822 | **0.941** | 1.000 | **0.857** |

Reply quality, judged only on the 10 messages gold-labelled `auto_handle`
(the only case where the draft is actually the thing that would be sent):
mean offline-judge overall score **4.0 / 5**, mean word-overlap-F1 vs. a
reference reply **0.409**. (Both computed with no LLM API key configured in
this environment -- see the "misleading" section below for what that does
and doesn't tell you.)

**Reading these numbers honestly:**

- Intent accuracy is *identical* between "simple" and "agent" (0.941, same
  0.822 macro-F1) because both use the literal same classifier function --
  the agent's improvement over "simple" is entirely in what happens *after*
  classification, not in classification itself. If someone skims only the
  intent-accuracy row, this system looks like it has no edge over the
  simplest possible baseline. It does, just not there.
- The real separation is escalation recall: 0.286 (simple) vs. 0.857
  (agent). The simple baseline's blind spot is exactly the golden set's
  contrast case 119299 (an account/access-code request) that its own naive
  intent-only rule *does* catch, but it completely misses every
  high-frustration message (119253, 119268, 119272) because it has no tone
  signal at all -- those get the same boilerplate as a calm complaint.
- Escalate precision is 1.000 for both simple and agent on this golden set.
  With only 17 examples and roughly a 40/60 auto/escalate split, that's not
  strong evidence the agent never over-escalates -- see below.

## Failure analysis: top 5

All five below are documented in-line in `eval/golden_set.csv`'s
`labeler_notes` or `DECISION_LOG.md`, with the specific tweet ids.

1. **No thread/conversation context at classification time (119249).**
   "Me too am suffering, hope the can find a solution" is meaningless in
   isolation -- it only reads as a performance complaint once you know it's
   a reply in the same thread as 119250's "ios too slow on iphone6." The
   classifier sees each tweet independently and correctly has no idea what
   to do with it, landing on `other` -> escalate. The decision is arguably
   still right (escalate), but for the wrong reason ("unclassified junk"
   instead of "needs the parent tweet"), and a *positive* classification
   result here would be pure luck.

2. **No signal for "customer already tried the documented fix" (119290).**
   The customer explicitly says they followed the brand's own help-page
   steps (disabled background apps and Location Services) and it didn't
   help. The system's grounding score is *high* (0.375, well above the
   auto-handle threshold) because the message shares vocabulary with other
   battery complaints -- but the historically-grounded reply is exactly the
   generic "which iOS version are you on" question the customer has already
   effectively answered. High grounding score here is actively misleading:
   it measures topical similarity, not "is this reply actually useful given
   what the customer already told us."

3. **Retrieval sometimes grounds on the wrong exemplar (119326).** "My apps
   stop working... freezes every five minutes" retrieved a historical
   exchange about Apple Music pausing during WhatsApp use as its best
   match, and the drafted reply asks "does the music pause?" -- a question
   about a topic the customer never raised. TF-IDF over a 12-example corpus
   has no real semantic understanding; a handful of shared generic tokens
   (this after already stripping `@mentions`/URLs, see Decision Log #7) is
   enough to misrank. My offline rubric judge scored this reply's
   "grounded" dimension a 2/5 (some credit for asking *a* diagnostic
   question); my own hand-scoring gave it a 1/5, because "asks a plausible-
   sounding but topically wrong question" is worse than "asks a generic
   question," and the rubric's overlap-based check can't tell those apart.

4. **Single-path drafting is unsafe for intents the historical corpus
   doesn't represent (119291, found via evaluation, now fixed).** Before the
   fix documented in Decision Log #9, a "problem solved, thanks!" message
   got drafted a diagnostic troubleshooting question, because the retrieval
   corpus in this sample contains zero resolved/positive exchanges to
   ground against. This is now special-cased, but the general lesson is
   more important than the specific fix: any intent under-represented (or
   absent) in the historical grounding corpus is a landmine for a
   retrieval-grounded drafter, and there's no guarantee there isn't another
   one of these in an intent this dataset happens not to exercise.

5. **The rule-based escalation policy is stricter than what the brand
   actually, historically did (119253, 119268, 119272, 119299).** Real 2017
   AppleSupport sent the same generic boilerplate to profane, angry
   messages and to an account-security request as it did to a calm bug
   report. If "how often does the agent match the brand's real historical
   behavior" were the metric, these four would count as *misses*. I built
   the golden labels and the escalation policy around a different, explicit
   judgment call (Decision Log #5) -- but that's exactly the kind of
   assumption a reviewer should be able to see and disagree with, not one
   that should hide inside an aggregate accuracy number.

## What is misleading about my headline number

This section is mandatory, and honestly the most important one, so three
separate things, not one:

**"94.1% escalation accuracy" is inflated by the eval set being small and
built by the same person who built the thresholds.** 17 examples, hand-
labelled by me, using rules I also wrote. The escalation-decision thresholds
(frustration word list, 2-point cutoff, 0.12 grounding cutoff) were tuned
by looking at exactly these 17 messages. That's the textbook setup for a
model (or a rule set) to look better on its own eval set than it would on a
genuinely new one. I mitigated this in two small, honest ways -- leave-one-
out retrieval (Decision Log #8) so grounding scores aren't trivially perfect,
and deliberately including two "hard case" gold labels that disagree with
what my own rules output (Decision Log #14, items 2 and 3 in the failure
analysis) -- but two countermeasures on a 17-example, single-annotator eval
set do not make 94.1% a number you should trust the way you'd trust it on a
held-out set of hundreds of examples from other reviewers.

**The judge-agreement statistics look almost contradictory, and that's
itself informative, not a bug.** `eval/judge_agreement.py` reports Pearson
r = -0.171 and Spearman r = 0.029 (essentially no correlation) but 90% of
judge scores land within 0.5 points of my hand score, and mean absolute
error is only 0.375 on a roughly 1-5 scale. Both are true at once because
every reply in this sample scored in a narrow band (3.5-5.0) -- with that
little spread, tiny absolute differences reorder the ranking, which
correlation is sensitive to and MAE isn't. If I only reported the
correlation number, this system's judge would look essentially useless at
tracking quality; if I only reported MAE/within-0.5, it would look nearly
perfect. Neither framing alone is honest; the real finding is narrower and
more useful than either: the offline judge's biggest, most legible failure
is that it doesn't know a resolved/positive message shouldn't get a
diagnostic question (119291, a 1.25-point miss, by far the largest single
disagreement in the set) -- everything else it does is within noise of my
own hand-scoring.

**"Judge agrees with a human" here means "agrees with the one person who
built the whole pipeline."** I hand-scored the same 10 replies the automated
judge scored, and I built the judge, the golden set, and the escalation
policy. That is a real, useful sanity check (it caught the 119291 gap, and
the 119326 grounded-on-wrong-topic gap) but it is not evidence of
inter-annotator agreement with an independent reviewer, which is what the
number would need to mean anything at production scale. Decision Log #10
flags this in code as well as here.

## What I'd do with one more week

In rough priority order: (1) get the actual Kaggle dataset (a machine with
unrestricted egress, or downloading it locally and shipping a larger
`data/twcs_full_sample.csv`), re-derive the intent taxonomy against a much
larger AppleSupport slice, and rebuild the golden set as a genuine 150-250-
example random sample with sampling documented, ideally split so at least
some of it is never looked at while tuning; (2) get a second, independent
human labeler for both the golden set and the judge-agreement subset, since
right now both are me; (3) replace TF-IDF retrieval with a small sentence-
embedding bi-encoder once there's enough data for it to matter, specifically
to fix failure mode #3 (grounding on a topically wrong exemplar); (4) add a
"customer already tried standard troubleshooting" detector (failure mode
#2) -- likely a short LLM classification pass over the message rather than
a keyword rule, since the phrasing varies a lot; (5) support multi-label
intents (Decision Log #15) so compound messages like 119270 don't force a
single lossy label; (6) build a small regression test suite that pins known
bugs found during this build (the `"app"`/`"applesupport"` substring
collision, the `@mention` retrieval-leakage, and the positive-feedback
drafting bug) so they can't silently reappear.
