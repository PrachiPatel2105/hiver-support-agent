"""
Evaluation harness. Run: `python3 eval/run_eval.py`

Loads eval/golden_set.csv (17 hand-labelled AppleSupport messages -- see
README.md "Data note" and golden_set.csv's own notes for why 17, not
150-250), runs three systems against it:

  1. trivial baseline  -- majority-class intent, always auto_handle
  2. simple baseline    -- keyword intent + naive "auto_handle unless
                            account_access/other" decision (no tone or
                            grounding checks at all)
  3. full agent          -- src/agent.py (intent + retrieval-grounded draft +
                            frustration/grounding-aware escalation)

and reports intent accuracy/F1, escalation-decision accuracy/precision/recall,
and reply-quality judge scores (auto-handled messages only, since that's the
only case where the agent's draft is the thing actually sent).

Leave-one-out retrieval (see agent.build_index) is used throughout so no
result benefits from a query matching itself in the grounding corpus.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from load_data import load_tweets, exchanges_for_brand
from agent import build_index, run_agent, decide, draft_reply
from intents import rule_based_intent, trivial_intent, INTENTS
from judge import judge_reply

DATA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "twcs_sample.csv")
GOLDEN_PATH = str(Path(__file__).resolve().parent / "golden_set.csv")
BRAND = "AppleSupport"


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_reference_reply(tweet_id: str, gold_reply_reference: str, exchange_by_customer_id: dict) -> str:
    if gold_reply_reference != "captured":
        return gold_reply_reference
    exch = exchange_by_customer_id.get(tweet_id)
    return exch.brand_tweet.text if exch else ""


def word_overlap_f1(a: str, b: str) -> float:
    """Cheap ROUGE-1-F1 stand-in: no extra dependency, same spirit (unigram
    overlap between draft and reference reply)."""
    import re
    wa = set(re.findall(r"[a-z']+", a.lower()))
    wb = set(re.findall(r"[a-z']+", b.lower()))
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    p = inter / len(wa)
    r = inter / len(wb)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def simple_baseline_decide(intent: str) -> tuple[str, str]:
    """Baseline 2: routes purely on intent, no tone/grounding awareness."""
    if intent in {"account_access", "other"}:
        return "escalate", "intent-only rule: account_access/other always escalate"
    return "auto_handle", "intent-only rule: everything else auto-handles"


def run() -> dict:
    golden = load_golden()
    tweets = load_tweets(DATA_PATH)
    all_exchanges = exchanges_for_brand(tweets, BRAND)
    exchange_by_customer_id = {e.customer_tweet.tweet_id: e for e in all_exchanges}

    gold_intents, gold_decisions = [], []
    trivial_intents, trivial_decisions = [], []
    simple_intents, simple_decisions = [], []
    agent_intents, agent_decisions = [], []

    reply_rows = []  # for judge scoring, only where gold decision == auto_handle

    for row in golden:
        tid = row["tweet_id"]
        text = row["customer_text"]
        gi, gd = row["gold_intent"], row["gold_decision"]
        gold_intents.append(gi)
        gold_decisions.append(gd)

        # Baseline 1: trivial
        ti, _ = trivial_intent(text)
        trivial_intents.append(ti)
        trivial_decisions.append("auto_handle")  # trivial baseline never escalates

        # Baseline 2: simple (rule intent + intent-only decision, no retrieval draft)
        si, _ = rule_based_intent(text)
        simple_intents.append(si)
        sd, _ = simple_baseline_decide(si)
        simple_decisions.append(sd)

        # Full agent (leave-one-out retrieval)
        idx = build_index(DATA_PATH, BRAND, exclude_customer_tweet_ids={tid})
        out = run_agent(text, idx)
        agent_intents.append(out.intent)
        agent_decisions.append(out.decision)

        reference = resolve_reference_reply(tid, row["gold_reply_reference"], exchange_by_customer_id)
        reply_rows.append({
            "tweet_id": tid,
            "text": text,
            "gold_decision": gd,
            "reference_reply": reference,
            "agent_reply": out.reply_draft,
            "agent_decision": out.decision,
            "llm_backend": out.llm_backend,
        })

    def clf_report(name, preds, labels_key):
        gold = gold_intents if labels_key == "intent" else gold_decisions
        acc = accuracy_score(gold, preds)
        f1 = f1_score(gold, preds, average="macro", zero_division=0)
        return {"system": name, "task": labels_key, "accuracy": round(acc, 3), "macro_f1": round(f1, 3)}

    # Per-intent breakdown for the full agent (intent labels are identical for
    # agent and simple since they share the same classifier, so we report once).
    intent_labels = sorted(set(gold_intents))
    per_intent = {}
    for label in intent_labels:
        p = precision_score(gold_intents, agent_intents, labels=[label], average="macro", zero_division=0)
        r = recall_score(gold_intents, agent_intents, labels=[label], average="macro", zero_division=0)
        f = f1_score(gold_intents, agent_intents, labels=[label], average="macro", zero_division=0)
        support = gold_intents.count(label)
        per_intent[label] = {
            "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(f, 3), "support": support,
        }

    # Confusion matrix (agent)
    all_labels = sorted(set(gold_intents + agent_intents))
    cm = confusion_matrix(gold_intents, agent_intents, labels=all_labels).tolist()

    results = {
        "classification": [],
        "per_intent_breakdown": per_intent,
        "confusion_matrix": {"labels": all_labels, "matrix": cm},
        "escalation": [],
        "reply_quality": {},
        "reply_rows": reply_rows,
    }

    for name, preds in [("trivial", trivial_intents), ("simple", simple_intents), ("agent", agent_intents)]:
        results["classification"].append(clf_report(name, preds, "intent"))

    for name, preds in [("trivial", trivial_decisions), ("simple", simple_decisions), ("agent", agent_decisions)]:
        acc = accuracy_score(gold_decisions, preds)
        prec = precision_score(gold_decisions, preds, pos_label="escalate", zero_division=0)
        rec = recall_score(gold_decisions, preds, pos_label="escalate", zero_division=0)
        results["escalation"].append({
            "system": name, "accuracy": round(acc, 3),
            "escalate_precision": round(prec, 3), "escalate_recall": round(rec, 3),
        })

    # Reply quality: judge every reply the FULL AGENT drafted for messages
    # gold-labelled auto_handle (the only case where a drafted reply would
    # actually be sent to the customer), plus word-overlap vs. reference.
    judge_scores, overlaps = [], []
    for r in reply_rows:
        if r["gold_decision"] != "auto_handle":
            continue
        js = judge_reply(r["text"], r["agent_reply"])
        overlap = word_overlap_f1(r["agent_reply"], r["reference_reply"]) if r["reference_reply"] else None
        judge_scores.append(js.overall)
        if overlap is not None:
            overlaps.append(overlap)
        r["judge_overall"] = round(js.overall, 2)
        r["judge_backend"] = js.backend
        r["reply_vs_reference_word_f1"] = round(overlap, 3) if overlap is not None else None

    results["reply_quality"] = {
        "n_scored": len(judge_scores),
        "mean_judge_overall": round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None,
        "mean_word_overlap_f1_vs_reference": round(sum(overlaps) / len(overlaps), 3) if overlaps else None,
    }

    return results


if __name__ == "__main__":
    results = run()
    out_path = Path(__file__).resolve().parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Intent classification ===")
    for r in results["classification"]:
        print(f"  {r['system']:8s} acc={r['accuracy']:.3f}  macro_f1={r['macro_f1']:.3f}")

    print("\n=== Per-intent breakdown (agent / simple — same classifier) ===")
    print(f"  {'intent':<22s} {'precision':>9} {'recall':>7} {'f1':>7} {'support':>8}")
    print("  " + "-" * 57)
    for intent, m in results["per_intent_breakdown"].items():
        print(f"  {intent:<22s} {m['precision']:>9.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} {m['support']:>8d}")

    print("\n=== Confusion matrix (agent) ===")
    labels = results["confusion_matrix"]["labels"]
    cm_data = results["confusion_matrix"]["matrix"]
    col_w = max(len(l) for l in labels) + 2
    header = "pred →".ljust(col_w) + "".join(l[:col_w-1].ljust(col_w) for l in labels)
    print("  " + header)
    for label, row in zip(labels, cm_data):
        print("  " + label[:col_w-1].ljust(col_w) + "".join(str(v).ljust(col_w) for v in row))

    print("\n=== Escalation decision ===")
    for r in results["escalation"]:
        print(f"  {r['system']:8s} acc={r['accuracy']:.3f}  escalate_precision={r['escalate_precision']:.3f}  escalate_recall={r['escalate_recall']:.3f}")

    print("\n=== Reply quality (auto_handle messages only) ===")
    rq = results["reply_quality"]
    print(f"  n={rq['n_scored']}  mean_judge_overall={rq['mean_judge_overall']}  mean_word_overlap_f1_vs_reference={rq['mean_word_overlap_f1_vs_reference']}")
    print(f"\nFull results written to {out_path}")
