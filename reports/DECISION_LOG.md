# Decision log

Non-obvious decisions made while building this, and why. Numbered for
cross-referencing from code comments and REPORT.md.

1. **Ran against the 93-row provided sample, not the full Kaggle dataset.**
   This sandboxed environment has no outbound network access to Kaggle or
   Hugging Face (confirmed by trying: both `kaggle.com` and
   `huggingface.co` are blocked by the environment's egress proxy). Rather
   than fabricate a larger synthetic dataset to hit the letter of "150-250
   examples," I built an honestly-scaled pipeline against what was actually
   available and documented the gap explicitly everywhere it matters
   (README, golden set, report). A fabricated dataset would have produced
   numbers that looked more complete and were actually less trustworthy --
   exactly backwards from what this assignment is testing.

2. **Picked AppleSupport as the brand.** It has the most outbound messages
   in the 93-row sample (13, vs. 8 for the next two), which maximizes the
   amount of real historical grounding data available -- important when the
   whole dataset is this small.

3. **LLM calls are behind a pluggable backend that degrades gracefully with
   no API key.** A grader running this without network access or an API key
   configured still gets a fully working, structured pipeline (retrieval-
   adapted replies and a rubric-based judge instead of free-generated text
   and an LLM judge). The alternative -- requiring a key to run at all --
   would fail the "reproduce results in under 15 minutes" bar for anyone
   without one already set up.

4. **Intent taxonomy has exactly 5 classes plus an `other` catch-all**,
   read directly off the 17 real AppleSupport-directed messages rather than
   designed top-down. `other` always escalates by design (see #6) --
   an unclassified message is exactly the case where auto-handling is
   riskiest, so the taxonomy doesn't need to be exhaustive to be safe.

5. **Escalation policy is intentionally stricter than what the brand
   actually did historically.** Real 2017 AppleSupport sent the identical
   generic "thanks for reaching out, DM us" boilerplate to messages
   containing profanity (119268, 119253) and to an account-security request
   (119299) alike. This agent's golden labels and decision rules escalate
   profane/high-frustration messages and always escalate identity-touching
   ones, even though a naive "does this match what the brand actually did"
   metric would score that as a miss. I judged customer-facing tone risk and
   account-security risk as real product requirements the brand's own 2017
   social-media playbook didn't optimize for, not gaps in my modeling.
   Directly discussed in REPORT.md's baseline comparison.

6. **`decide()` is a small set of documented constants (a frustration word
   list, an emoji set, a grounding-score threshold), not a model.**
   Requirement #3 asks for a *stated reason* for every escalation decision.
   A stated reason is only worth something if a human reviewer can trace it
   back to something inspectable -- a black-box classifier's confidence
   score isn't that. Keeping the escalation logic legible was a deliberate
   trade against a fancier learned policy that would likely fit this tiny
   dataset worse anyway (17 examples is nowhere near enough to fit anything
   with real capacity).

7. **Found and fixed a real matching bug: `@AppleSupport` mentions were
   polluting both retrieval and intent classification.** The retrieval
   index and the `"app"` keyword rule both originally ran on raw tweet text.
   Since almost every customer message contains the literal substring
   `"applesupport"`, (a) TF-IDF cosine similarity was inflated between
   *any* two messages that both mention the brand (an unrelated
   notification/UI-bug report scored 0.20 similarity to a "fix this update"
   message, purely from shared `"applesupport"` tokens), and (b) the intent
   keyword `"app"` (meant to catch "my app crashed") fired on literally
   every message, because `"app"` is a substring of `"applesupport"`. Fixed
   by stripping `@handles` and URLs before both indexing/querying
   (`src/text_utils.py`) and by requiring a word-boundary match specifically
   for the `"app"` keyword (`src/intents.py`). This is exactly the kind of
   silent, plausible-looking bug that a single trivial test case (`"@brand
   totally unrelated message"` scoring low similarity to everything) would
   have caught immediately -- worth calling out as a real "how would you
   catch this in CI" answer.

8. **Evaluation uses leave-one-out retrieval.** 13 of the 17 golden-set
   messages are *also* one of the 12 historical exchanges the retrieval
   index is built from (they're the same real tweets). Querying the index
   without excluding the query's own historical exchange first retrieves
   itself at cosine similarity 1.0, making grounding (and therefore the
   auto-handle decision) look artificially perfect. `agent.build_index`
   takes `exclude_customer_tweet_ids` and `eval/run_eval.py` always passes
   the current query's own id, so every reported grounding score reflects
   what the agent would see on a genuinely new message.

9. **Found and fixed a reply-drafting bug via evaluation, not by
   inspection: positive-feedback messages were getting a diagnostic
   question as a reply.** The retrieval corpus in this sample contains zero
   "thanks / already resolved" exchanges (every captured historical reply
   is a response to an open complaint), so grounding a "problem solved!"
   message against it retrieved a troubleshooting reply, and the offline
   drafter dutifully adapted it into "When did this start happening?" --
   sent to a customer who just said their problem is fixed. This surfaced
   from `eval/run_eval.py`'s output, not from reading the code, which is
   itself evidence for why the harness matters more than eyeballing the
   pipeline. Fixed by special-casing `positive_feedback` in `agent.run_agent`
   to skip retrieval-grounded drafting entirely (see REPORT.md failure
   analysis #4 for the general pattern this points at: single-path grounded
   drafting is unsafe for intents the historical corpus doesn't represent).

10. **Judge-human agreement is measured against my own hand-scores, not an
    independent annotator's**, because building this alone in a sandboxed
    environment ruled out a second labeler. I've flagged this explicitly
    (`eval/judge_agreement.py` docstring, REPORT.md) as an upper bound on
    real inter-rater agreement rather than a substitute for it -- the honest
    answer to "how well does your judge agree with a human" here is "we
    don't fully know yet," and pretending otherwise would be exactly the
    kind of misleading headline number the assignment's mandatory section
    asks about.

11. **Reply quality is only judged for gold-`auto_handle` messages.** For an
    escalated message the agent's draft is, at best, a suggested starting
    point for a human -- scoring it as if it were the thing actually sent
    to the customer would conflate two different quality bars.

12. **`positive_feedback` requires the literal phrase `"love apple"`, not
    just `"love"`.** 119326 ("apps stop working... Love the new update!!!!")
    is sarcastic. A looser `"love"` keyword would misclassify it as positive
    feedback and, worse, auto-close a real bug report. This is a narrow
    fix for one observed case, not a general sarcasm detector -- flagged
    honestly as a gap in REPORT.md rather than oversold as solved.

13. **Word-overlap-F1 against a reference reply is reported as a secondary,
    weak signal, not a headline metric.** It's cheap and dependency-free,
    but two replies can be equally good with almost no shared vocabulary
    (see REPORT.md) -- it's included for triangulation alongside the judge
    score, not as a stand-in for it.

14. **The golden set intentionally includes "hard cases" where my own gold
    label disagrees with what the automated system would output** (119249,
    119290 -- see their `labeler_notes`), rather than only including cases
    where I was confident the system would get it right. An eval set built
    by whoever also built the thresholds is prone to grading its own
    homework leniently; deliberately seeding known disagreements is a small
    countermeasure, discussed further in REPORT.md.

15. **Multi-label intents were explicitly scoped out.** Several messages
    (e.g. 119270: battery + crashes + a downgrade request) genuinely carry
    more than one intent. Single-label was chosen to keep the taxonomy, the
    golden set, and the metrics simple enough to build and check by hand at
    this scale; flagged in REPORT.md as the first thing to change with more
    time and more data.
