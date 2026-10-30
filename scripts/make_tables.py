"""Aggregate outputs/results/summary.csv into report tables and figures.

Run from repo root (after scripts/score_all.py):  python scripts/make_tables.py [--split test]
Writes: outputs/results/table_<split>.md, outputs/results/table_<split>.csv,
        outputs/figures/recall_vs_overtriage_<split>.png, outputs/figures/script_recall_<split>.png
Rows are grouped by system + training scripts; seeds are summarised as mean +/- sd (sd needs >= 2 seeds).
"""
import argparse
import csv
import statistics as st
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

METRICS = [("em_recall", "Emerg. recall"), ("over_triage", "Over-triage"), ("under_triage", "Under-triage"),
           ("macro_f1", "Macro-F1"), ("script_gap", "Script gap")]
SCRIPT_COLS = [("recall_en", "English"), ("recall_deva", "Devanagari"), ("recall_hinglish", "Hinglish")]
ORDER = ["Keyword baseline", "Qwen prompted, 0-shot", "Qwen prompted, 8-shot",
         "MuRIL (train: all)", "MuRIL (train: en)", "MuRIL (train: deva)", "MuRIL (train: hinglish)",
         "QLoRA gate (train: all)", "QLoRA gate (train: en)", "QLoRA gate (train: deva)",
         "QLoRA gate (train: hinglish)"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def group_name(r):
    s, ts = r["system"], r["train_scripts"]
    if s.startswith("muril"):
        return f"MuRIL (train: {ts})"
    if s.startswith("qlora"):
        return f"QLoRA gate (train: {ts})"
    if s == "v0_zeroshot":
        return "Qwen prompted, 0-shot"
    if s == "v0_fewshot8":
        return "Qwen prompted, 8-shot"
    if s == "keyword":
        return "Keyword baseline"
    return s


def vals(rows, col):
    return [v for v in (num(r[col]) for r in rows) if v is not None]


def ms(v):
    if not v:
        return "-"
    if len(v) < 2:
        return f"{st.mean(v):.3f}"
    return f"{st.mean(v):.3f} ± {st.stdev(v):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    a = ap.parse_args()
    src = ROOT / "outputs/results/summary.csv"
    rows = [r for r in csv.DictReader(open(src, encoding="utf-8")) if r["split"] == a.split]
    if not rows:
        sys.exit(f"no rows for split {a.split} in {src}")
    by = defaultdict(list)
    for r in rows:
        by[group_name(r)].append(r)
    groups = [g for g in ORDER if g in by] + sorted(g for g in by if g not in ORDER)

    header = ["System", "runs"] + [m[1] for m in METRICS] + [s[1] + " recall" for s in SCRIPT_COLS]
    table = [header]
    for g in groups:
        table.append([g, str(len(by[g]))] + [ms(vals(by[g], c)) for c, _ in METRICS]
                     + [ms(vals(by[g], c)) for c, _ in SCRIPT_COLS])

    out_dir = ROOT / "outputs/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"table_{a.split}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(table)
    md = ["| " + " | ".join(table[0]) + " |", "|" + "---|" * len(table[0])]
    md += ["| " + " | ".join(r) + " |" for r in table[1:]]
    (out_dir / f"table_{a.split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    fig_dir = ROOT / "outputs/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: every run = one point (recall vs over-triage): shows recall bought by over-triage
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
    for i, g in enumerate(groups):
        xs, ys = vals(by[g], "over_triage"), vals(by[g], "em_recall")
        ax.scatter(xs, ys, s=55, marker=markers[i % len(markers)], label=g, alpha=0.85)
    ax.set_xlabel("Over-triage rate (non-emergency predicted EMERGENCY), lower is better")
    ax.set_ylabel("Emergency recall, higher is better")
    ax.set_title(f"Recall vs over-triage per run ({a.split})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"recall_vs_overtriage_{a.split}.png", dpi=200)
    plt.close(fig)

    # Figure 2: emergency recall per script (mean over seeds, error bar = sd)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.26
    x = np.arange(len(groups))
    for j, (col, lab) in enumerate(SCRIPT_COLS):
        means, errs = [], []
        for g in groups:
            v = vals(by[g], col)
            means.append(st.mean(v) if v else 0.0)
            errs.append(st.stdev(v) if len(v) > 1 else 0.0)
        ax.bar(x + (j - 1) * w, means, w, yerr=errs, capsize=2, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(g, 14) for g in groups], fontsize=7)
    ax.set_ylabel("Emergency recall")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Emergency recall by script ({a.split})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"script_recall_{a.split}.png", dpi=200)
    plt.close(fig)
    print(f"\nsaved tables to outputs/results/table_{a.split}.* and figures to outputs/figures/")


if __name__ == "__main__":
    main()
