"""Temperature-scale one run's probabilities: fit T on DEV, report ECE/NLL before and after on dev and test.

Run from repo root:  python scripts/calibrate.py --run outputs/raw/qlora_all_s13 [--boot 200]
Writes outputs/results/calibration_<run>.json and calibrated predictions to outputs/calibrated/<run>/
(preds_dev.jsonl, preds_test.jsonl). It does NOT touch outputs/raw or outputs/results/summary.csv.

Temperature scaling leaves each prediction's argmax unchanged (macro-F1 and the default decisions do not move);
ECE and threshold-based analyses (e.g. the H1 matched over-triage budgets) can change. Dev has only 45 cases, so
the fitted T is noisy: read the bootstrap interval before trusting it.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import kio  # noqa: E402
from swasthya.calibration import (apply_temperature, bootstrap_temperature, calibration_report,  # noqa: E402
                                  fit_temperature)


def load(run, split):
    gold = kio.read_jsonl(ROOT / f"data/d3_triage/rendered/d3_{split}.jsonl")
    preds = {r["id"]: r for r in kio.read_jsonl(run / f"preds_{split}.jsonl")}
    missing = [g["id"] for g in gold if g["id"] not in preds or "probs" not in preds[g["id"]]]
    if missing:
        sys.exit(f"{run.name} {split}: {len(missing)} gold ids lack probabilities")
    return gold, np.array([preds[g["id"]]["probs"] for g in gold])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    run = Path(a.run)
    if not run.is_absolute():
        run = ROOT / run
    g_dev, p_dev = load(run, "dev")
    g_te, p_te = load(run, "test")
    y_dev, y_te = [g["label"] for g in g_dev], [g["label"] for g in g_te]
    T = fit_temperature(p_dev, y_dev)
    lo, hi = bootstrap_temperature(p_dev, y_dev, [g["case_id"] for g in g_dev], n=a.boot)
    rep = calibration_report(p_dev, y_dev, p_te, y_te, T)
    rep.update({"run": run.name, "temperature_ci95_dev_cases": [lo, hi], "n_dev_cases": len({g["case_id"] for g in g_dev})})
    (ROOT / "outputs/results").mkdir(parents=True, exist_ok=True)
    (ROOT / f"outputs/results/calibration_{run.name}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    out = ROOT / "outputs/calibrated" / run.name
    for split, g, p in (("dev", g_dev, p_dev), ("test", g_te, p_te)):
        cal = apply_temperature(p, T)
        kio.write_jsonl(out / f"preds_{split}.jsonl", kio.pred_rows(g, cal.tolist()))
    if (run / "config.json").exists():
        shutil.copy(run / "config.json", out / "config.json")
    print(f"{run.name}: T = {T:.2f}  (95% over dev cases: {lo:.2f} - {hi:.2f}; T>1 means the model was over-confident)")
    for split in ("dev", "test"):
        print(f"  ECE {split}: {rep[f'ece_{split}_before']:.3f} -> {rep[f'ece_{split}_after']:.3f}")
    print(f"  NLL test: {rep['nll_test_before']:.3f} -> {rep['nll_test_after']:.3f}")
    print(f"saved outputs/results/calibration_{run.name}.json and outputs/calibrated/{run.name}/")


if __name__ == "__main__":
    main()
