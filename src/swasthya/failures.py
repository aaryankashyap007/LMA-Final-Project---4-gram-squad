"""Error analysis of a triage run: classify every error, summarise, and pick candidate cases to read.

Pure functions over gold rows (d3_*.jsonl) and predictions ({"id", "pred", "probs"}). The selection of the 6-8
cases for the report is deterministic and documented in `pick_candidates`; reading and explaining them is a human
job, so scripts/failure_analysis.py only prints them with their context.
"""
from collections import Counter, defaultdict

from .metrics import SEVERITY
from .schema import LABELS, SCRIPTS


def error_type(gold, pred):
    if gold == pred:
        return "correct"
    if gold == "EMERGENCY":
        return "missed_emergency"
    if pred == "EMERGENCY":
        return "false_alarm"
    if gold == "INSUFFICIENT":
        return "committed_on_vague"          # model gave a definite level to a message that needs questions
    if pred == "INSUFFICIENT":
        return "wrongly_insufficient"        # asked for more when the message was clear enough
    return "under_triage" if SEVERITY[pred] < SEVERITY[gold] else "over_triage"


def join(gold_rows, preds):
    """One record per gold row with prediction, error type and the probability of the predicted class."""
    out = []
    for g in gold_rows:
        p = preds[g["id"]]
        probs = p.get("probs")
        out.append({**g, "pred": p["pred"], "probs": probs, "etype": error_type(g["label"], p["pred"]),
                    "conf": probs[LABELS.index(p["pred"])] if probs else None,
                    "p_emergency": probs[0] if probs else None})
    return out


def summarise(rows):
    conf = {g: Counter() for g in LABELS}
    for r in rows:
        conf[r["label"]][r["pred"]] += 1
    errs = [r for r in rows if r["etype"] != "correct"]
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    wrong_scripts = Counter(sum(x["etype"] != "correct" for x in v) for v in by_case.values())
    pair_fail = []
    pairs = defaultdict(list)
    for r in rows:
        if r.get("pair_id"):
            pairs[(r["pair_id"], r["script"])].append(r)
    for (pid, s), m in sorted(pairs.items()):
        if len(m) == 2 and m[0]["label"] != m[1]["label"] and m[0]["pred"] == m[1]["pred"]:
            pair_fail.append((pid, s, m[0]["case_id"], m[1]["case_id"], m[0]["pred"]))
    return {"n": len(rows), "n_errors": len(errs), "confusion": conf,
            "by_type": Counter(r["etype"] for r in errs),
            "by_script": {s: sum(1 for r in errs if r["script"] == s) for s in SCRIPTS},
            "by_category": Counter(r["category"] for r in errs),
            "cases_by_n_wrong_scripts": dict(sorted(wrong_scripts.items())), "pair_failures": pair_fail}


def _conf(r):
    return r["conf"] if r["conf"] is not None else 0.0


def pick_candidates(rows, k=8):
    """Deterministic shortlist of up to k errors that cover different failure modes, one reason each:
      1-3  most confident missed emergency in each script (a missed emergency is the safety failure)
      4    false alarm with the highest P(EMERGENCY)
      5    script-specific failure: a case wrong in exactly one script (prefers gold EMERGENCY)
      6    minimal-pair failure: both members of a pair got the same prediction
      7    most confident URGENT->ROUTINE under-triage
      8    most confident confusion involving INSUFFICIENT
    then fills up to k with the most confident remaining errors. Different cases are preferred over repeats. If the
    best example of a failure mode is a row already chosen for another mode, that row is tagged with both reasons
    instead of the mode being dropped."""
    errs = [r for r in rows if r["etype"] != "correct"]
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    chosen, used_ids, used_cases = [], set(), set()

    def take(reason, cands, ordered=False):
        if not ordered:
            cands = sorted(cands, key=lambda r: (-_conf(r), r["id"]))
        for fresh_case_only in (True, False):          # prefer a case not shown yet, else allow a repeat
            for r in cands:
                if r["id"] in used_ids or (fresh_case_only and r["case_id"] in used_cases):
                    continue
                chosen.append([reason, r])
                used_ids.add(r["id"])
                used_cases.add(r["case_id"])
                return
        for entry in chosen:                            # every example is already shown: tag the best one
            if cands and entry[1]["id"] == cands[0]["id"]:
                entry[0] += f"; {reason}"
                return

    for s in SCRIPTS:
        take(f"missed emergency ({s})", [r for r in errs if r["etype"] == "missed_emergency" and r["script"] == s])
    take("false alarm", sorted([r for r in errs if r["etype"] == "false_alarm"],
                               key=lambda r: -(r["p_emergency"] or 0.0)))
    single = sorted((r for r in errs if sum(x["etype"] != "correct" for x in by_case[r["case_id"]]) == 1),
                    key=lambda r: (r["label"] != "EMERGENCY", -_conf(r), r["id"]))     # emergencies first
    take("script-specific failure", single, ordered=True)
    pairs = defaultdict(list)
    for r in rows:
        if r.get("pair_id"):
            pairs[(r["pair_id"], r["script"])].append(r)
    flat = [x for m in pairs.values() if len(m) == 2 and m[0]["label"] != m[1]["label"]
            and m[0]["pred"] == m[1]["pred"] for x in m if x["etype"] != "correct"]
    take("minimal pair did not flip", flat)
    take("under-triage URGENT->ROUTINE", [r for r in errs if r["label"] == "URGENT" and r["pred"] == "ROUTINE"])
    take("INSUFFICIENT confusion", [r for r in errs if "INSUFFICIENT" in (r["label"], r["pred"])
                                    and r["etype"] != "missed_emergency"])
    for r in sorted(errs, key=lambda r: (-_conf(r), r["id"])):
        if len(chosen) >= k:
            break
        if r["id"] not in used_ids:
            chosen.append(["most confident remaining error", r])
            used_ids.add(r["id"])
    return [(why, r) for why, r in chosen[:k]]
