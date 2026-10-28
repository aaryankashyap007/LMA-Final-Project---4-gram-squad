"""Validate the human-written D3 EVAL set (d3_cases.csv); optionally export dev/test jsonl.
Run from repo root:  python scripts/validate_d3.py [--export]
(Training data is handled by scripts/prepare_train.py.)
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import d3  # noqa: E402


def main():
    f = ROOT / "data/d3_triage/human_cases/d3_cases.csv"
    if not f.exists():
        print("Missing", f)
        return 1
    rows = d3.load_csv(f)
    print(f"{f.name}: {len(rows)} cases")
    errors, warnings = d3.validate_rows(rows)
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"\nTotal {len(rows)} cases | labels {dict(Counter(r['label'] for r in rows))}")
    print(f"categories {dict(Counter(r['category'] for r in rows))}")
    rep = d3.parity_report(rows)
    for s, (rate, n) in rep.items():
        print(f"parity[{s}]: {'n/a' if rate is None else f'{rate:.2%}'} (n={n})")
    if errors:
        print(f"\n{len(errors)} error(s) - fix before exporting.")
        return 1
    if "--export" in sys.argv:
        rows = d3.assign_splits(rows)
        counts = d3.export_jsonl(rows, ROOT / "data/d3_triage/rendered")
        print("exported to data/d3_triage/rendered/  (split,label)->renderings:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
