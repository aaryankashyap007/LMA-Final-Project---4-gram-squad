"""Mutation test of the action guard (see src/swasthya/mutation.py).

  python scripts/mutation_test.py                      # all mutants, attacks + unit tests
  python scripts/mutation_test.py --no_tests           # attacks only (a few seconds)
  python scripts/mutation_test.py --mutant no_ownership --out outputs/results/mutation.json

For every mutant: how many attack episodes (scripted agent that obeys the injected request, guard-on arm) are
still harmed, which families, and how many unit tests fail with the mutant applied. A mutant that neither the
attacks nor the tests notice SURVIVES and points at a missing test.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya.mutation import MUTANTS, apply, attack_summary, dumps, run_probes, worst_case_by_family  # noqa: E402
from swasthya.scenario import load_scenarios  # noqa: E402

FILES = ["data/d4_clinic/scenarios_adversarial_v0.jsonl", "data/d4_clinic/scenarios_adversarial_v1.jsonl"]


def run_tests(name):
    env = dict(os.environ, SWASTHYA_MUTANT=name, PYTHONPATH=str(ROOT / "src"))
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "--ignore=tests/test_mutation.py"], cwd=ROOT, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    m = re.search(r"(\d+) failed", out)
    failed = int(m.group(1)) if m else (0 if p.returncode == 0 else -1)   # -1: crashed before running
    return failed, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", action="append")
    ap.add_argument("--no_tests", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    scenarios = [s for f in FILES for s in load_scenarios(ROOT / f)]
    names = a.mutant or list(MUTANTS)
    results = {}
    base = attack_summary(worst_case_by_family(scenarios))
    assert base["harmed"] == 0, f"the unmutated guard is already harmed: {base}"
    assert not any(run_probes().values()), "the unmutated guard is harmed by a probe"
    rows = []
    for n in names:
        undo = apply(n)
        try:
            att = attack_summary(worst_case_by_family(scenarios))
            att["probes_harmed"] = sorted(k for k, v in run_probes().items() if v)
        finally:
            undo()
        failed = None if a.no_tests else run_tests(n)[0]
        results[n] = {"description": MUTANTS[n][0], "attacks": att, "tests_failed": failed}
        killed_by = []
        if att["harmed"] or att["probes_harmed"]:
            killed_by.append("attacks")
        if failed:
            killed_by.append("tests")
        status = "+".join(killed_by) if killed_by else ("not run" if failed is None else "SURVIVED")
        rows.append((n, f"{att['harmed']}/{att['episodes']}", len(att["probes_harmed"]),
                     "-" if failed is None else failed, status))
        print(f"{n}: harmed {att['harmed']}/{att['episodes']}, probes {att['probes_harmed']}, tests failed {failed}, {status}", flush=True)
    print("\nunmutated guard: 0 harmed of", base["episodes"], "attack episodes\n")
    w = [max(len(str(x)) for x in col) for col in zip(("mutant", "attacks harmed", "probes harmed", "tests failed", "result"),
                                                       *rows)]
    for r in [("mutant", "attacks harmed", "probes harmed", "tests failed", "result")] + rows:
        print("  ".join(str(x).ljust(n) for x, n in zip(r, w)))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(dumps({"baseline": base, "mutants": results}), encoding="utf-8")
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
