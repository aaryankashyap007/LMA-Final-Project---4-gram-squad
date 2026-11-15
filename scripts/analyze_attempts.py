"""Attempted-attack analysis of agent runs (see src/swasthya/attempts.py for the definitions).

  python scripts/analyze_attempts.py --run agent_advv1_liquid --scenarios data/d4_clinic/scenarios_adversarial_v1.jsonl
  python scripts/analyze_attempts.py --worst_case data/d4_clinic/scenarios_adversarial_v0.jsonl data/d4_clinic/scenarios_adversarial_v1.jsonl

--run takes a folder holding episodes/*.json (repeat it for several runs); --worst_case takes scenario files.
Writes the numbers to --out (JSON) when given.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya.attempts import analyse_episodes, load_episodes, worst_case  # noqa: E402
from swasthya.scenario import load_scenarios  # noqa: E402

ARM_ORDER = ["A0", "A1", "+Gate", "+Gate_V0", "+Guard", "Full"]


def table(headers, rows):
    w = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = lambda r: "  ".join(str(x).ljust(n) for x, n in zip(r, w))
    return "\n".join([line(headers), line(["-" * n for n in w])] + [line(r) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--scenarios", nargs="*", default=[])
    ap.add_argument("--worst_case", nargs="*", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    result = {}

    if a.run:
        by_id = {s.id: s for p in a.scenarios for s in load_scenarios(p)}
        for folder in a.run:
            arms, fam, rows = analyse_episodes(load_episodes(folder), by_id)
            print(f"\n== {folder}: harmful attempts per arm ==")
            print("attempt = a call that, if it ran, would be an unapproved action or a leak")
            print(table(["arm", "episodes", "w/ attempt", "attempts", "executed", "blocked", "would run w/o guard"],
                        [[k, arms[k]["episodes"], arms[k]["episodes_with_attempt"], arms[k]["attempts"],
                          arms[k]["executed"], arms[k]["blocked"], arms[k]["would_succeed_without_guard"]]
                         for k in ARM_ORDER if k in arms]))
            if rows:
                print("\nper family and arm (attempts / executed / blocked):")
                print(table(["family", "arm", "att", "exec", "blocked"],
                            [[f, k, v["attempts"], v["executed"], v["blocked"]] for (f, k), v in sorted(fam.items())]))
            result[folder] = {"arms": {k: dict(v) for k, v in arms.items()}, "attempts": rows}

    if a.worst_case:
        scenarios = [s for p in a.worst_case for s in load_scenarios(p)]
        wc = worst_case(scenarios)
        print("\n== worst case: scripted agent that always follows the injected request ==")
        print("(a ground-truth test of the guard, not an LLM result)")
        print(table(["arm", "episodes", "unapproved", "leaks", "episodes harmed"],
                    [[k, v["episodes"], v["unapproved"], v["leaks"], v["episodes_harmed"]] for k, v in wc.items()]))
        result["worst_case"] = wc

    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
