"""H1: emergency recall at a FIXED over-triage rate, with the threshold chosen on the dev split.

Run from repo root (after scripts/score_all.py):  python scripts/h1_matched_overtriage.py

For every run that trained on all scripts (or is a prompted system) and has probabilities:
  1. on DEV pick the smallest threshold t on P(EMERGENCY) such that dev over-triage <= alpha
     (over-triage = share of non-emergency cases predicted EMERGENCY);
  2. on TEST predict EMERGENCY iff P(EMERGENCY) >= t and report emergency recall and the real test over-triage;
  3. compare G1 (qlora) with each prompted system using a PAIRED bootstrap over test CASES
     (the 3 script renderings of one case are resampled together).
Writes outputs/results/h1_matched_overtriage.csv and outputs/results/h1_paired_diffs.csv
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import kio  # noqa: E402

ALPHAS = [0.05, 0.10, 0.20]
PRIMARY = 0.10
N_BOOT = 1000


def load_split(run_dir, split):
    gold = kio.read_jsonl(ROOT / f"data/d3_triage/rendered/d3_{split}.jsonl")
    preds = {r["id"]: r for r in kio.read_jsonl(run_dir / f"preds_{split}.jsonl")}
    p_em = np.array([preds[g["id"]]["probs"][0] for g in gold])      # index 0 = EMERGENCY (schema.LABELS)
    is_em = np.array([g["label"] == "EMERGENCY" for g in gold])
    cases = np.array([g["case_id"] for g in gold])
    return p_em, is_em, cases


def pick_threshold(p_em, is_em, alpha):
    """Smallest t with dev over-triage <= alpha (most permissive threshold that respects the budget)."""
    neg = p_em[~is_em]
    for t in np.unique(np.concatenate([p_em, [1.0 + 1e-9]])):
        if (neg >= t).mean() <= alpha:
            return float(t)
    return 1.0 + 1e-9


def make_boots(cases, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(cases)
    members = [np.where(cases == u)[0] for u in uniq]
    return [np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n)]


def ms(v):
    if len(v) < 2:
        return f"{v[0]:.3f}"
    return f"{st.mean(v):.3f} ± {st.stdev(v):.3f}"


def main():
    runs = []
    for d in sorted((ROOT / "outputs/raw").iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        if not (d / "preds_dev.jsonl").exists() or not (d / "preds_test.jsonl").exists():
            continue
        cfgp = d / "config.json"
        cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
        if str(cfg.get("train_scripts", "all")) not in ("all", "-"):
            continue
        runs.append((d.name, cfg.get("system", d.name), d))
    if not runs:
        sys.exit("no runs found in outputs/raw")

    rows, brec, boots, em_te = [], {}, None, None
    for name, system, d in runs:
        p_dev, em_dev, _ = load_split(d, "dev")
        p_te, em_te, cases = load_split(d, "test")
        if boots is None:
            boots = make_boots(cases)
        for a in ALPHAS:
            t = pick_threshold(p_dev, em_dev, a)
            pred = p_te >= t
            recall = float(pred[em_te].mean())
            over_test = float(pred[~em_te].mean())
            over_dev = float((p_dev[~em_dev] >= t).mean())
            bv = np.array([pred[b][em_te[b]].mean() for b in boots])
            brec[(name, a)] = bv
            rows.append({"run": name, "system": system, "alpha": a, "threshold": round(t, 6),
                         "dev_over": round(over_dev, 4), "test_recall": round(recall, 4),
                         "ci_lo": round(float(np.percentile(bv, 2.5)), 4),
                         "ci_hi": round(float(np.percentile(bv, 97.5)), 4), "test_over": round(over_test, 4)})
    out = ROOT / "outputs/results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "h1_matched_overtriage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for a in ALPHAS:
        print(f"\n=== alpha = {a:.2f}  (threshold set on DEV so that dev over-triage <= {a:.2f}) ===")
        print(f"{'system':<18}{'runs':>5}   {'test emerg. recall':<20}{'test over-triage':<20}")
        by = defaultdict(list)
        for r in rows:
            if r["alpha"] == a:
                by[r["system"]].append(r)
        for s, rr in by.items():
            print(f"{s:<18}{len(rr):>5}   {ms([r['test_recall'] for r in rr]):<20}{ms([r['test_over'] for r in rr]):<20}")

    diffs = []
    g1 = [(n, s) for n, s, _ in runs if s == "qlora"]
    prompted = [(n, s) for n, s, _ in runs if s.startswith("v0_")]
    print("\n=== Paired difference in test emergency recall: G1 (qlora) minus prompted gate ===")
    print("(95% CI from a bootstrap over test cases; '*' = CI excludes 0)")
    for a in ALPHAS:
        for gn, _ in g1:
            for pn, ps in prompted:
                dv = brec[(gn, a)] - brec[(pn, a)]
                point = next(r["test_recall"] for r in rows if r["run"] == gn and r["alpha"] == a) - \
                    next(r["test_recall"] for r in rows if r["run"] == pn and r["alpha"] == a)
                lo, hi = float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))
                star = "*" if (lo > 0 or hi < 0) else " "
                diffs.append({"alpha": a, "g1_run": gn, "prompted_run": pn, "diff": round(point, 4),
                              "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "excludes_zero": star == "*"})
                if a == PRIMARY:
                    print(f"alpha {a:.2f}  {gn:<16} vs {pn:<12} diff {point:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    with open(out / "h1_paired_diffs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(diffs[0].keys()))
        w.writeheader()
        w.writerows(diffs)
    print("saved outputs/results/h1_matched_overtriage.csv")


if __name__ == "__main__":
    main()
