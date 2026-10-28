"""Validate d3_train.csv, check overlap with dev/test, export data/d3_triage/rendered/d3_train.jsonl.

Run from repo root (after validate_d3.py --export):  python scripts/prepare_train.py [--drop-sim 0.8]
Train cases whose text is a near-copy (char 5-gram Jaccard >= drop-sim, in ANY script) of a dev/test
rendering are dropped from the export (whole case, so scripts stay parallel).
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import d3, kio  # noqa: E402
from swasthya.keyword_baseline import norm  # noqa: E402
from swasthya.schema import SCRIPTS  # noqa: E402

REND = ROOT / "data/d3_triage/rendered"


def grams(t, n=5):
    t = f" {norm(t)} "
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/d3_triage/human_cases/d3_train.csv"))
    ap.add_argument("--drop-sim", type=float, default=0.8)
    a = ap.parse_args()

    rows = d3.load_csv(a.csv)
    print(f"{Path(a.csv).name}: {len(rows)} cases")
    errors, warnings = d3.validate_rows(rows)
    for r in rows:
        if not r["case_id"].upper().startswith("T"):
            warnings.append(f"{r['case_id']}: id does not start with T")
    for w in warnings[:20]:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    if errors:
        print(f"\n{len(errors)} error(s). Fix d3_train.csv (or ask the writer) and re-run.")
        return 1
    print("labels", dict(Counter(r["label"] for r in rows)))
    print("categories", dict(Counter(r["category"] for r in rows)))

    gold = []
    for sp in ("dev", "test"):
        p = REND / f"d3_{sp}.jsonl"
        if not p.exists():
            print("Missing", p, "- run: python scripts/validate_d3.py --export")
            return 1
        gold += kio.read_jsonl(p)
    gold_grams = {s: [(g["id"], grams(g["text"])) for g in gold if g["script"] == s] for s in SCRIPTS}

    case_max, matches = {}, []
    for r in rows:
        for s in SCRIPTS:
            tg = grams(r[s])
            best_id, best = max(((gid, jac(tg, gg)) for gid, gg in gold_grams[s]), key=lambda x: x[1])
            case_max[r["case_id"]] = max(case_max.get(r["case_id"], 0.0), best)
            if best >= 0.4:
                matches.append((r["case_id"], s, round(best, 3), best_id))
    sims = sorted(case_max.values())
    print(f"\nTrain-vs-eval similarity (max over scripts): mean {sum(sims) / len(sims):.3f} "
          f"max {sims[-1]:.3f} | >=0.5: {sum(x >= 0.5 for x in sims)} | >={a.drop_sim}: {sum(x >= a.drop_sim for x in sims)}")
    gold_cues = {tuple(sorted(set(g["cues"]))) for g in gold}
    same_cues = sum(tuple(d3.parse_cues(r["cues"])) in gold_cues for r in rows)
    print(f"train cases whose exact cue-set also occurs in dev/test: {same_cues}/{len(rows)}")
    out_csv = ROOT / "outputs/results/train_eval_similarity.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train_case", "script", "jaccard5", "closest_eval_id"])
        w.writerows(sorted(matches, key=lambda m: -m[2]))
    print("saved", out_csv.relative_to(ROOT), f"({len(matches)} pairs with sim >= 0.4)")

    drop = {c for c, m in case_max.items() if m >= a.drop_sim}
    if drop:
        print("DROPPED (near-copies of dev/test):", sorted(drop))
    out = []
    for r in rows:
        if r["case_id"] in drop:
            continue
        for s in SCRIPTS:
            out.append({"id": f"{r['case_id']}|{s}", "case_id": r["case_id"], "script": s, "text": r[s],
                        "label": r["label"], "category": r["category"], "pair_id": None, "split": "train",
                        "cues": d3.parse_cues(r["cues"])})
    kept = len(rows) - len(drop)
    if kept < 0.8 * len(rows):
        print(f"\nERROR: {len(drop)} of {len(rows)} train cases would be dropped as near-copies of dev/test.")
        print("That is too many: look at outputs/results/train_eval_similarity.csv, talk to the writer, or raise --drop-sim.")
        return 1
    kio.write_jsonl(REND / "d3_train.jsonl", out)
    print(f"\nexported {len(out)} renderings ({kept} cases) -> data/d3_triage/rendered/d3_train.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
