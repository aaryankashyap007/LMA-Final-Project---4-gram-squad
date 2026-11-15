"""Statistical analysis of the 2x2 ablation (gate on/off x guard on/off). Definitions are in src/swasthya/stats.py.

  python scripts/analyze_ablation.py \\
      --run v0_nano=outputs/raw/agent_adv_nano::data/d4_clinic/scenarios_adversarial_v0.jsonl \\
      --run v1_nano=outputs/raw/agent_advv1_nano::data/d4_clinic/scenarios_adversarial_v1.jsonl

Each --run is  label=episodes_folder::scenario_file.  Runs are pooled; pairs are always within a run and scenario,
so different models or sets never mix inside a comparison. Use --out for the JSON and --md for a markdown table.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya.attempts import load_episodes  # noqa: E402
from swasthya.scenario import load_scenarios  # noqa: E402
from swasthya.stats import (arm_table, build_records, compare, interaction, primary_family)  # noqa: E402

ORDER = ["A0", "A1", "+Gate", "+Gate_V0", "+Guard", "Full"]


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table(headers, rows):
    w = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = lambda r: "  ".join(str(x).ljust(n) for x, n in zip(r, w))
    return "\n".join([line(headers), line(["-" * n for n in w])] + [line(r) for r in rows])


def parse_run(spec):
    label, rest = spec.split("=", 1)
    folder, scen = rest.split("::", 1)
    return label, folder, scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=folder::scenarios.jsonl")
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--out")
    ap.add_argument("--md")
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        label, folder, scen = parse_run(spec)
        runs.append((label, load_episodes(folder), {s.id: s for s in load_scenarios(scen)}))
    records, excluded = build_records(runs)
    attack = [r for r in records.values() if r["attack"]]
    print(f"{len(runs)} runs, {len(records)} scenario-runs ({len(attack)} attack, "
          f"{sum(r['utility'] for r in records.values())} utility), excluded flagged episodes: {excluded or 'none'}")

    result = {"excluded": excluded}
    tab = arm_table(records)
    print("\n== harm per arm, attack scenarios (exact 95% interval) ==")
    print(table(["arm", "episodes", "harmed", "rate", "95% CI", "harmful attempts"],
                [[k, v["n"], v["harm"], f"{v['harm'] / v['n']:.3f}" if v["n"] else "-",
                  f"{v['harm_ci'][0]:.3f}-{v['harm_ci'][1]:.3f}", v["attempt"]]
                 for k in ORDER if k in tab for v in [tab[k]]]))
    result["arms"] = tab

    fam = primary_family(records, a.reps)
    print("\n== the four comparisons of the 2x2 on harm (difference = first arm minus second) ==")
    print(table(["comparison", "arms", "pairs", "rates", "diff [95% CI]", "only 1st/2nd", "p", "p Holm", "families p Holm"],
                [[k, f"{c['a']} vs {c['b']}", c["pairs"], f"{c['rate_a']:.3f} / {c['rate_b']:.3f}",
                  f"{c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]", f"{c['only_a']}/{c['only_b']}",
                  fmt_p(c["p"]), fmt_p(c["p_holm"]), f"{fmt_p(c['p_family'])} -> {fmt_p(c['p_family_holm'])}"]
                 for k, c in fam.items()]))
    result["primary"] = fam

    ix = interaction(records, "harm", "attack", a.reps)
    if ix:
        print(f"\ninteraction (guard effect with the gate off minus with the gate on), harm: {ix['value']:+.3f} "
              f"[{ix['ci'][0]:+.3f}, {ix['ci'][1]:+.3f}] over {ix['pairs']} scenarios "
              f"(guard effect {ix['guard_effect_gate_off']:+.3f} gate off, {ix['guard_effect_gate_on']:+.3f} gate on)")
    result["interaction"] = ix

    att = [compare(records, x, y, "attempt", "attack", a.reps) for x, y in (("A1", "+Guard"), ("+Gate", "Full"))]
    att = [c for c in att if c]
    if att:
        print("\n== propensity: harmful attempts (guard on versus off, same gate) ==")
        print(table(["arms", "pairs", "rates", "diff [95% CI]", "p"],
                    [[f"{c['a']} vs {c['b']}", c["pairs"], f"{c['rate_a']:.3f} / {c['rate_b']:.3f}",
                      f"{c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]", fmt_p(c["p"])] for c in att]))
    result["attempts"] = att

    util = [compare(records, x, y, "success", "utility", a.reps)
            for x, y in (("A1", "+Guard"), ("+Gate", "Full"), ("A1", "+Gate"), ("+Guard", "Full"))]
    util = [c for c in util if c]
    if util:
        print("\n== task success on benign and legitimate scenarios (cost of the layers) ==")
        print(table(["arms", "pairs", "success", "diff [95% CI]", "only 1st/2nd", "p"],
                    [[f"{c['a']} vs {c['b']}", c["pairs"], f"{c['rate_a']:.3f} / {c['rate_b']:.3f}",
                      f"{c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]", f"{c['only_a']}/{c['only_b']}",
                      fmt_p(c["p"])] for c in util]))
    result["utility"] = util

    print("\nRead with care: scripts of one family are near copies (see the family-level p); with a handful of events "
          "no test has power, so a large p is not evidence of no effect; one run per scenario at temperature 0.")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        print("wrote", a.out)
    if a.md:
        lines = ["| comparison | arms | pairs | diff [95% CI] | only 1st/2nd | p (Holm) | family p (Holm) |",
                 "|---|---|---|---|---|---|---|"]
        for k, c in fam.items():
            lines.append(f"| {k} | {c['a']} vs {c['b']} | {c['pairs']} | {c['diff']:+.3f} [{c['ci'][0]:+.3f}, "
                         f"{c['ci'][1]:+.3f}] | {c['only_a']}/{c['only_b']} | {fmt_p(c['p_holm'])} | "
                         f"{fmt_p(c['p_family_holm'])} |")
        Path(a.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("wrote", a.md)


if __name__ == "__main__":
    main()
