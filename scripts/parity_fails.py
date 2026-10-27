"""List rows that failed the second-annotator parity check.
Run from repo root:  python scripts/parity_fails.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import d3

rows = d3.load_csv(ROOT / "data/d3_triage/human_cases/d3_cases.csv")
bad = 0
for r in rows:
    if not any(r.get(f"a2_present_{s}") for s in ("en", "deva", "hinglish")):
        continue  # not checked yet
    gold = set(d3.parse_cues(r["cues"]))
    extra = r.get("a2_extra", "").lower() == "yes"
    problems = []
    for s in ("en", "deva", "hinglish"):
        got = set(d3.parse_cues(r.get(f"a2_present_{s}", "")))
        if got != gold:
            problems.append(f"  {s}: checker saw {sorted(got)}  | missing {sorted(gold-got)} extra-in-checker {sorted(got-gold)}")
    if extra or problems:
        bad += 1
        print(f"{r['case_id']}  label={r['label']}  gold cues={sorted(gold)}")
        if extra:
            print("  a2_extra = yes  (some text has a medical detail not in cues)")
        print("\n".join(problems))
        print(f"  EN: {r['en']}\n  DEVA: {r['deva']}\n  HINGLISH: {r['hinglish']}\n")
print(f"{bad} row(s) to fix")
