"""
Evidence of how well the (offline, rubric-based) judge agrees with a human.

Methodology note (be honest about this -- it's exactly the kind of thing the
assignment's mandatory "what's misleading about my headline number" section
asks for): the "human" here is the assignment author hand-scoring the same
10 replies the automated judge scored, on the same 4-axis rubric, written
down BEFORE looking at the judge's numbers (see eval/human_judge_scores.csv
notes, which call out specific disagreements). This is not an independent
annotator, so treat the agreement number as an upper bound on real
inter-rater agreement, not a substitute for it -- see DECISION_LOG.md #10 and
REPORT.md.

Run: `python3 eval/judge_agreement.py` (run_eval.py must have been run first
so eval_results.json exists).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent


def main() -> None:
    with open(HERE / "human_judge_scores.csv", newline="", encoding="utf-8") as f:
        human = {row["tweet_id"]: float(row["human_overall"]) for row in csv.DictReader(f)}

    with open(HERE / "eval_results.json") as f:
        results = json.load(f)

    pairs = []
    for row in results["reply_rows"]:
        tid = row["tweet_id"]
        if tid in human and row.get("judge_overall") is not None:
            pairs.append((tid, human[tid], row["judge_overall"]))

    if not pairs:
        print("No overlapping scored rows found -- run eval/run_eval.py first.")
        return

    human_scores = np.array([p[1] for p in pairs])
    judge_scores = np.array([p[2] for p in pairs])

    mae = float(np.mean(np.abs(human_scores - judge_scores)))
    pearson_r, _ = pearsonr(human_scores, judge_scores)
    spearman_r, _ = spearmanr(human_scores, judge_scores)
    # "within half a point" agreement -- a more forgiving, interpretable
    # threshold given the small sample and 4-point-ish scale.
    within_half = float(np.mean(np.abs(human_scores - judge_scores) <= 0.5))

    print(f"n = {len(pairs)}")
    print(f"Mean absolute error (human vs. offline judge): {mae:.3f}")
    print(f"Pearson r:  {pearson_r:.3f}")
    print(f"Spearman r: {spearman_r:.3f}")
    print(f"Fraction within 0.5 points: {within_half:.2f}")
    print()
    print("Per-item (tweet_id, human, judge, abs_diff):")
    for tid, h, j in pairs:
        print(f"  {tid}: human={h:.2f}  judge={j:.2f}  |diff|={abs(h - j):.2f}")

    biggest = max(pairs, key=lambda p: abs(p[1] - p[2]))
    print(f"\nLargest disagreement: tweet {biggest[0]} (human={biggest[1]}, judge={biggest[2]}).")
    print("See eval/human_judge_scores.csv notes for why -- the offline judge's")
    print("actionable check (presence of '?' or 'DM') doesn't know that a")
    print("resolved/positive message shouldn't ask a diagnostic question at all.")

    out = {
        "n": len(pairs), "mae": round(mae, 3), "pearson_r": round(float(pearson_r), 3),
        "spearman_r": round(float(spearman_r), 3), "fraction_within_0.5": round(within_half, 3),
        "pairs": [{"tweet_id": t, "human": h, "judge": j} for t, h, j in pairs],
    }
    with open(HERE / "judge_agreement_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
