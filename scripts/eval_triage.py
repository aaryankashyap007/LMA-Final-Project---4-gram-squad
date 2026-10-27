"""Score a triage system on D3 (dev or test).

Keyword baseline:
  python scripts/eval_triage.py --data data/d3_triage/rendered/d3_test.jsonl --keyword --name b0_keyword_test
Any other system (preds.jsonl from Kaggle):
  python scripts/eval_triage.py --data data/d3_triage/rendered/d3_test.jsonl \
      --preds outputs/raw/muril_s42/preds_test.jsonl --name b1_muril_s42_test
Writes outputs/results/<name>.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import evaluate as ev  # noqa: E402
from swasthya.keyword_baseline import predict  # noqa: E402


def f(x):
    return "  n/a" if x is None or x != x else f"{x:.3f}"


def row(name, b):
    lo, hi = b["emergency_recall_ci"]
    return (f"{name:<9} n={b['n']:<4} nE={b['n_emergency']:<3} EmRecall {f(b['emergency_recall'])} "
            f"[{f(lo)},{f(hi)}]  over {f(b['over_triage'])}  under {f(b['under_triage'])}  "
            f"macroF1 {f(b['macro_f1'])}  ECE {f(b.get('ece'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds")
    ap.add_argument("--keyword", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    gold = ev.load_jsonl(a.data)
    if a.keyword:
        preds = {g["id"]: {"id": g["id"], "pred": predict(g["text"])} for g in gold}
    elif a.preds:
        preds = {r["id"]: r for r in ev.load_jsonl(a.preds)}
    else:
        ap.error("give --preds or --keyword")
    res = ev.evaluate(gold, preds, n_boot=a.boot)
    out = ROOT / "outputs/results" / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(row("overall", res["overall"]))
    for s, b in res["by_script"].items():
        print(row(s, b))
    print(f"cross-script emergency-recall gap: {f(res['cross_script_recall_gap'])}")
    for s, d in res["minimal_pairs"].items():
        print(f"pairs[{s}]: n={d['n_pairs']} flip_acc={f(d['flip_acc'])} both_correct={f(d['both_correct'])}")
    print("saved", out)


if __name__ == "__main__":
    main()
