"""Score every preds file under outputs/raw/ (+ the keyword baseline) and build one summary table.

Run from repo root after copying Kaggle outputs into outputs/raw/:  python scripts/score_all.py [--boot 1000]
Writes outputs/results/<run>_<split>.json for every run and outputs/results/summary.csv
Runs whose preds do not cover every gold id (e.g. smoke tests) are skipped.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import evaluate as ev  # noqa: E402
from swasthya.keyword_baseline import predict  # noqa: E402


def f3(x):
    return "" if x is None or x != x else round(float(x), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--raw", default="outputs/raw")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    a = ap.parse_args()
    gold = {s: ev.load_jsonl(ROOT / f"data/d3_triage/rendered/d3_{s}.jsonl") for s in a.splits}
    jobs = []
    for sp, g in gold.items():
        jobs.append(("b0_keyword", {"system": "keyword", "train_scripts": "-", "seed": "-", "shots": "-"}, sp,
                     {x["id"]: {"pred": predict(x["text"])} for x in g}))
    for pf in sorted((ROOT / a.raw).rglob("preds_*.jsonl")):
        sp = pf.stem.split("_", 1)[1]
        if sp not in gold or pf.parent.name.startswith("smoke"):
            continue
        cfgp = pf.parent / "config.json"
        cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {"system": pf.parent.name}
        jobs.append((pf.parent.name, cfg, sp, {r["id"]: r for r in ev.load_jsonl(pf)}))
    rows = []
    for run, cfg, sp, preds in jobs:
        try:
            res = ev.evaluate(gold[sp], preds, n_boot=a.boot)
        except ValueError as e:
            print("SKIP", run, sp, "-", str(e)[:90])
            continue
        (ROOT / "outputs/results").mkdir(parents=True, exist_ok=True)
        (ROOT / f"outputs/results/{run}_{sp}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        o, lo_hi = res["overall"], res["overall"]["emergency_recall_ci"]
        rec, mp = res["emergency_recall_by_script"], res["minimal_pairs"]
        rows.append({
            "run": run, "system": cfg.get("system"), "train_scripts": cfg.get("train_scripts"), "seed": cfg.get("seed"),
            "shots": cfg.get("shots"), "split": sp, "n": o["n"], "em_recall": f3(o["emergency_recall"]),
            "em_ci_lo": f3(lo_hi[0]), "em_ci_hi": f3(lo_hi[1]), "over_triage": f3(o["over_triage"]),
            "under_triage": f3(o["under_triage"]), "macro_f1": f3(o["macro_f1"]), "ece": f3(o.get("ece")),
            "script_gap": f3(res["cross_script_recall_gap"]),
            **{f"recall_{s}": f3(rec.get(s)) for s in ("en", "deva", "hinglish")},
            **{f"flip_{s}": f3(mp[s]["flip_acc"]) for s in ("en", "deva", "hinglish")}})
    out = ROOT / "outputs/results/summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'run':<26}{'split':<6}{'emRec':>7}{'[CI]':>16}{'over':>7}{'under':>7}{'mF1':>7}{'gap':>7}")
    for r in sorted(rows, key=lambda r: (r["split"], r["run"])):
        ci = f"[{r['em_ci_lo']},{r['em_ci_hi']}]"
        print(f"{r['run']:<26}{r['split']:<6}{r['em_recall']:>7}{ci:>16}{r['over_triage']:>7}{r['under_triage']:>7}"
              f"{r['macro_f1']:>7}{r['script_gap']:>7}")
    print("\nsaved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
