# LLM-as-Judge Rubric — @AppleSupport Reply Quality

Used by `src/judge.py` to score every auto-handled reply on four axes.
When an LLM API key is available, this rubric is injected verbatim into the judge prompt.
When no key is set, `_rubric_judge()` implements the same four axes deterministically.
Human scores in `eval/human_judge_scores.csv` were assigned using this same rubric.

---

## Scoring scale

Each axis is scored **1–5** (integers only).  
**Overall = mean of the four axis scores** (may be non-integer).

| Score | Meaning |
|-------|---------|
| 5 | Excellent — fully meets the criterion |
| 4 | Good — mostly meets it with minor gaps |
| 3 | Acceptable — partial credit, noticeable gap |
| 2 | Poor — mostly fails the criterion |
| 1 | Bad — completely fails the criterion |

---

## Axis 1: Grounded (1–5)

**Question:** Does the reply reference something specific to *this* customer's actual stated problem, rather than a pure generic?

| Score | Definition | Example |
|-------|-----------|---------|
| 5 | Reply mentions ≥2 content words from the customer message (e.g. "battery", "iOS update", "WhatsApp") OR references the customer's specific symptom directly | Customer says "battery dies in 2 hrs"; reply asks "which iOS version are you on after your battery update?" |
| 4 | Reply references the customer's topic at a high level (e.g. "battery issue") without drilling into their specific detail | Customer says "wifi keeps dropping"; reply says "let's help with your connectivity issue" |
| 3 | Reply has one word of overlap with the customer message but is otherwise generic | Customer says "apps crashing"; reply says "we can help with your apps issue — DM us" |
| 2 | Reply is a generic support template with no reference to the customer's specific problem | "Thanks for reaching out to us. Send us a DM." — sent to any complaint |
| 1 | Reply is grounded on the **wrong** topic — references something the customer never mentioned | Customer says "phone freezes"; reply asks "does the music pause when you use WhatsApp?" |

**Offline rubric implementation:** counts shared content words (≥4 chars) between cleaned customer text and reply after stripping stopwords. Score 5 if ≥2 shared, 4 if 1, 2 if 0.  
**Known weakness:** cannot distinguish "no overlap" (score 2) from "wrong-topic overlap" (should be score 1). See tweet 119326 in `human_judge_scores.csv`.

---

## Axis 2: Actionable (1–5)

**Question:** Does the reply move the conversation forward — does it give the customer a concrete next step or ask a specific diagnostic question?

| Score | Definition | Example |
|-------|-----------|---------|
| 5 | Asks a specific, relevant diagnostic question AND/OR gives a concrete next step (e.g. "DM us your iOS version") | "Which version of iOS are you on? You can find it in Settings → General → About. DM us." |
| 4 | Asks a question or gives a next step, but it is somewhat generic | "Could you DM us more details?" |
| 3 | Vaguely implies a next step without making it concrete | "We'd like to look into this for you." |
| 2 | No question, no next step; reply just acknowledges the problem | "We're sorry to hear about your experience." |
| 1 | Actively discourages further engagement or is a dead end | "Please check our support page." (no DM invite, no question) |

**Offline rubric implementation:** score 5 if reply contains `?` OR the word `dm`; score 2 otherwise.  
**Known weakness:** a diagnostic question asked of a customer who already said their problem is solved (e.g. tweet 119291) should score 1, not 5 — the rubric cannot detect this. The human-labelled `positive_feedback` special-case in `agent.py` prevents the agent from generating this, but the rubric would still score it incorrectly if it appeared.

---

## Axis 3: Tone (1–5)

**Question:** Is the tone appropriate — does it acknowledge the customer's issue without being dismissive, robotic, or over-promising?

| Score | Definition | Example |
|-------|-----------|---------|
| 5 | Warm, specific acknowledgement; reads like a human wrote it for this customer | "Battery life is important and we hear you — let's dig into this in DM." |
| 4 | Has a standard acknowledgement phrase; feels professional but slightly templated | "Thanks for reaching out. We're happy to help." |
| 3 | Neutral and inoffensive; no real acknowledgement of the customer's frustration | "We can look into this." |
| 2 | Slightly robotic or could be read as dismissive; no warmth | A single sentence with no acknowledgement at all |
| 1 | Actively dismissive, sarcastic, or uses a tone the brand would never approve | "This is not our problem." |

**Offline rubric implementation:** score 4 if reply contains any of: `thanks`, `sorry`, `we'd like to help`, `we can help`, `that's what we love`; score 3 otherwise; floor to 2 if reply is under 15 characters.

---

## Axis 4: Safe (1–5)

**Question:** Does the reply avoid making unsupported promises, fabricating a fix timeline, or claiming the brand will do something it cannot guarantee?

| Score | Definition | Example of violation |
|-------|-----------|---------------------|
| 5 | No promises of outcome, no fabricated timeline, no guaranteed fix | "Let's take a look in DM" — commits only to investigation |
| 4 | Minor hedging language but no explicit promise | "We'll try to help" — weak but not a guarantee |
| 3 | Implies a likely fix without stating it explicitly | "We should be able to sort this out for you" |
| 2 | Makes an explicit but cautious promise | "We will fix this" |
| 1 | Fabricates a specific fix, refund, timeline, or policy the brand cannot know | "We'll issue a refund within 48 hours" or "The next update will fix this" |

**Offline rubric implementation:** score 2 if reply contains any of: `will fix`, `guaranteed`, `promise`, `definitely will`, `100%`; score 5 otherwise.

---

## Overall score

```
overall = (grounded + actionable + tone + safe) / 4
```

Scores in `eval/eval_results.json` (`judge_overall`) and `eval/human_judge_scores.csv` (`human_overall`) both use this formula.

---

## Calibration examples

| Tweet | Customer text (excerpt) | Good reply (score ~4.5) | Bad reply (score ~1.5) |
|-------|------------------------|------------------------|----------------------|
| 119291 | "problem solved, once again in love with Apple" | "That's what we love to hear! Thanks for sticking with us." | "When did this start happening? Which iOS version are you on?" |
| 119290 | "I already turned off all apps and location services but still losing 3% per typing session" | "Sorry the standard steps didn't help — since you've already tried that, let's dig deeper in DM." | "Have you tried turning off background app refresh? DM us." |
| 119326 | "My apps freeze every 5 mins! Love the new update!!!!" | "We'd like to help with the freezing. Which iOS version are you on? DM us." | "Does the music pause when you use WhatsApp? Let us know in DM." |

---

## Known rubric weaknesses (honest limitations)

1. **Wrong-topic grounding (Grounded axis):** The offline rubric scores any zero-overlap reply as 2, but a reply grounded on the wrong topic is worse than a generic one and should score 1. Requires semantic understanding to detect.
2. **Positive-feedback actionable (Actionable axis):** The `?`/`DM` check rewards diagnostic questions universally, but asking a diagnostic question of a resolved customer is actively wrong. The agent's `positive_feedback` special-case prevents this in practice, but the rubric itself cannot detect it.
3. **Same-annotator agreement:** Human scores in `eval/human_judge_scores.csv` were produced by the same person who wrote the rubric. True inter-annotator agreement with an independent reviewer has not been measured.
4. **Narrow score range:** In practice all 10 scored replies fell between 3.5 and 5.0 overall. This makes rank-order correlation statistics (Pearson, Spearman) near-zero even when absolute error (MAE) is small. See `eval/judge_agreement_results.json` and the interpretation block printed by `eval/judge_agreement.py`.
