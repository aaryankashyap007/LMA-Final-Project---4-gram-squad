"""Error analysis of one run on one split, written as markdown with candidate cases to read.

Run from repo root:
  python scripts/failure_analysis.py --preds outputs/raw/qlora_all_s13/preds_test.jsonl \
      --gold data/d3_triage/rendered/d3_test.jsonl --out docs/FAILURE_ANALYSIS_g1_s13.md [--k 8]
The tables are computed. The shortlist is picked by fixed rules (swasthya.failures.pick_candidates); someone must
read those cases and write what went wrong. The report should say how they were chosen and include hard ones.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import kio  # noqa: E402
from swasthya.failures import join, pick_candidates, summarise  # noqa: E402
from swasthya.schema import LABELS, SCRIPTS  # noqa: E402


def fmt_probs(p):
    return "n/a" if not p else " / ".join(f"{l[:3]} {x:.2f}" for l, x in zip(LABELS, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gold", default="data/d3_triage/rendered/d3_test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    gold = kio.read_jsonl(ROOT / a.gold)
    preds = {r["id"]: r for r in kio.read_jsonl(ROOT / a.preds)}
    rows = join(gold, preds)
    s = summarise(rows)
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["script"]] = r
    L = [f"# Failure analysis: `{a.preds}` on `{a.gold}`", "",
         f"{s['n_errors']} errors in {s['n']} renderings. Tables are computed; the shortlist below is chosen by fixed "
         "rules (see `swasthya.failures.pick_candidates`) and still needs a human reading.", "",
         "## Confusion matrix (rows = gold, columns = predicted)", "",
         "| gold \\ pred | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for g in LABELS:
        L.append(f"| **{g}** | " + " | ".join(str(s["confusion"][g][p]) for p in LABELS) + " |")
    L += ["", "## Errors by type", ""] + [f"- {k}: {v}" for k, v in s["by_type"].most_common()]
    L += ["", "## Errors by script", ""] + [f"- {k}: {v}" for k, v in s["by_script"].items()]
    L += ["", "## Errors by category", ""] + [f"- {k}: {v}" for k, v in s["by_category"].most_common()]
    L += ["", "## Cases by number of scripts predicted wrongly", "",
          "(0 = right in all three scripts; 3 = wrong in all three; 1-2 = a script-specific failure)", ""]
    L += [f"- {n} wrong: {c} cases" for n, c in s["cases_by_n_wrong_scripts"].items()]
    L += ["", "## Minimal pairs that did not flip", ""]
    L += [f"- pair {pid} ({sc}): cases {c1} and {c2} both predicted {p}" for pid, sc, c1, c2, p in s["pair_failures"]] \
        or ["- none"]
    L += ["", f"## Candidate cases (up to {a.k})", ""]
    for i, (reason, r) in enumerate(pick_candidates(rows, a.k), 1):
        others = ", ".join(f"{sc}: {by_case[r['case_id']][sc]['pred']}" for sc in SCRIPTS
                           if sc in by_case[r["case_id"]])
        L += [f"### {i}. {reason}: `{r['id']}`", "",
              f"- gold **{r['label']}**, predicted **{r['pred']}** ({r['etype']}), category {r['category']}",
              f"- probabilities: {fmt_probs(r['probs'])}",
              f"- same case in other scripts, predicted: {others}",
              f"- cues: {', '.join(r.get('cues', []))}",
              f"- text: {r['text']}", "- what went wrong: *(to be written after reading the case)*", ""]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"{s['n_errors']} errors; wrote {a.out}")


if __name__ == "__main__":
    main()
