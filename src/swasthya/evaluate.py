"""Evaluation harness shared by every triage system (keyword, MuRIL, prompted LLM, QLoRA gate).

Gold file: jsonl from d3.export_jsonl (id, case_id, script, text, label, pair_id, ...).
Preds file: jsonl, one line per gold id: {"id": "...", "pred": "EMERGENCY", "probs": [p_E, p_U, p_R, p_I]}
(probs optional; ordered as schema.LABELS).
"""
import json
from collections import defaultdict

from .schema import LABELS, SCRIPTS
from .metrics import (emergency_recall, over_triage_rate, under_triage_rate, macro_f1, ece,
                      bootstrap_ci, cross_script_recall_gap)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _block(y, p, groups, probs, n_boot):
    r = {"n": len(y), "n_emergency": sum(t == "EMERGENCY" for t in y),
         "emergency_recall": emergency_recall(y, p),
         "emergency_recall_ci": bootstrap_ci(emergency_recall, y, p, groups, n_boot),
         "over_triage": over_triage_rate(y, p), "under_triage": under_triage_rate(y, p),
         "macro_f1": macro_f1(y, p), "macro_f1_ci": bootstrap_ci(macro_f1, y, p, groups, n_boot)}
    if probs is not None:
        r["ece"] = ece(probs, y)
    return r


def pair_stats(gold, preds):
    """Minimal pairs: gold labels differ, so a good system's predictions must differ too."""
    by = defaultdict(list)
    for g in gold:
        if g.get("pair_id"):
            by[(g["pair_id"], g["script"])].append(g)
    out = {s: {"n_pairs": 0, "flip_acc": 0, "both_correct": 0} for s in SCRIPTS}
    for (pid, s), members in by.items():
        if len(members) != 2 or members[0]["label"] == members[1]["label"]:
            continue
        a, b = members
        pa, pb = preds[a["id"]]["pred"], preds[b["id"]]["pred"]
        out[s]["n_pairs"] += 1
        out[s]["flip_acc"] += int(pa != pb)
        out[s]["both_correct"] += int(pa == a["label"] and pb == b["label"])
    for s, d in out.items():
        n = d["n_pairs"]
        d["flip_acc"] = d["flip_acc"] / n if n else None
        d["both_correct"] = d["both_correct"] / n if n else None
    return out


def evaluate(gold, preds, n_boot=1000):
    missing = [g["id"] for g in gold if g["id"] not in preds]
    if missing:
        raise ValueError(f"{len(missing)} gold ids have no prediction, e.g. {missing[:3]}")
    bad = {preds[g["id"]]["pred"] for g in gold} - set(LABELS)
    if bad:
        raise ValueError(f"unknown predicted labels: {bad}")
    y = [g["label"] for g in gold]
    p = [preds[g["id"]]["pred"] for g in gold]
    groups = [g["case_id"] for g in gold]
    scripts = [g["script"] for g in gold]
    has_probs = all("probs" in preds[g["id"]] for g in gold)
    probs = [preds[g["id"]]["probs"] for g in gold] if has_probs else None
    res = {"overall": _block(y, p, groups, probs, n_boot), "by_script": {}}
    for s in SCRIPTS:
        idx = [i for i, x in enumerate(scripts) if x == s]
        if not idx:
            continue
        res["by_script"][s] = _block([y[i] for i in idx], [p[i] for i in idx],
                                     [groups[i] for i in idx],
                                     [probs[i] for i in idx] if probs else None, n_boot)
    gap, recs = cross_script_recall_gap(y, p, scripts)
    res["cross_script_recall_gap"] = gap
    res["emergency_recall_by_script"] = recs
    res["minimal_pairs"] = pair_stats(gold, preds)
    return res
