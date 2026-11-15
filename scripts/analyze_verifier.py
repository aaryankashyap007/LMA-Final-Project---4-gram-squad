"""Evaluate the medical-advice verifier (src/swasthya/verifier.py) three ways.

  python scripts/analyze_verifier.py \\
      --advice "advice=outputs/raw/agent_medadvice::data/d4_clinic/scenarios_medical_advice_v0.jsonl" \\
      --other "benign=outputs/raw/agent_benign" --other "v1=outputs/raw/agent_advv1_nano" --csv docs/review/verifier_live_replies.csv

1. Labelled replies (data/d4_clinic/verifier_cases.jsonl): recall on advice, false-block rate, by kind and script. The
   'easy' advice was written with the word lists in view, so its recall is optimistic; 'hard' advice is phrased with no
   listed word and shows what a word list cannot do.
2. Live replies to the medical-advice scenarios (--advice): per arm, how many replies the model gave, how many contain a
   hit, and how many were hand-offs or fallbacks without a model reply. Replies the verifier blocked are read from the
   episode's verifier log (the original text), so this works for runs made with --verifier on, shadow or off.
3. False blocks on ordinary replies (--other): the verifier is run over every model reply in runs that had no advice
   requests (booking, policy, hand-off...), and every hit is listed.
Writes a CSV of the live replies with a blank 'author label' column to check the unflagged ones by hand.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya.attempts import load_episodes  # noqa: E402
from swasthya.orchestrator import FALLBACK, TEMPLATES  # noqa: E402
from swasthya.scenario import load_scenarios  # noqa: E402
from swasthya.stats import clopper_pearson  # noqa: E402
from swasthya.verifier import SAFE_TEXT_EN, SAFE_TEXT_HI, Verifier  # noqa: E402

CASES = ROOT / "data/d4_clinic/verifier_cases.jsonl"
FIXED = {*TEMPLATES.values(), FALLBACK}
ORDER = ["A0", "A1", "+Gate", "+Gate_V0", "+Guard", "Full"]
V = Verifier()                         # version 1: refusal-aware (the one in use)
V0 = Verifier(refusal_aware=False)     # version 0: every listed word is a hit, as first built
PROV = ROOT / "docs/review/verifier_provisional_labels.json"


def table(headers, rows):
    w = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = lambda r: "  ".join(str(x).ljust(n) for x, n in zip(r, w))
    return "\n".join([line(headers), line(["-" * n for n in w])] + [line(r) for r in rows])


def ci(k, n):
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]" if n else "-"


def labelled():
    rows = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["hits"] = V.check(r["text"])
    adv = [r for r in rows if r["label"] == 1]
    ok = [r for r in rows if r["label"] == 0]
    print(f"== labelled replies: {len(adv)} with advice, {len(ok)} without ==")
    print("recall on all advice      ", ci(sum(bool(r["hits"]) for r in adv), len(adv)))
    for kind in ("easy", "hard"):
        sub = [r for r in adv if r["kind"] == kind]
        print(f"  recall on {kind:<5}          ", ci(sum(bool(r["hits"]) for r in sub), len(sub)))
    print("false blocks on all benign", ci(sum(bool(r["hits"]) for r in ok), len(ok)))
    for kind in ("refusal", "info", "tricky"):
        sub = [r for r in ok if r["kind"] == kind]
        print(f"  false blocks on {kind:<7}  ", ci(sum(bool(r["hits"]) for r in sub), len(sub)))
    print("\nby script (recall on advice / false blocks on benign):")
    print(table(["script", "recall", "false blocks"],
                [[s, ci(sum(bool(r["hits"]) for r in adv if r["script"] == s), sum(r["script"] == s for r in adv)),
                  ci(sum(bool(r["hits"]) for r in ok if r["script"] == s), sum(r["script"] == s for r in ok))]
                 for s in ("en", "hinglish", "deva")]))
    print("\nmissed advice:", *[f"  [{r['kind']}] {r['text']}" for r in adv if not r["hits"]], sep="\n")
    print("false blocks:", *[f"  [{r['kind']}] {r['text']}  -> {r['hits']}" for r in ok if r["hits"]], sep="\n")
    return rows


def model_replies(ep):
    """(turn, shown text, original text, blocked by the verifier) for each reply the language model wrote."""
    log = {v["turn"]: v for v in ep.get("verifier", [])}
    out = []
    for t in ep["transcript"]:
        if t["role"] != "assistant" or t["text"] in FIXED:
            continue
        v = log.get(t["turn"])
        original = v["original"] if v and v["blocked"] and t["text"] in (SAFE_TEXT_EN, SAFE_TEXT_HI) \
            or v and v["blocked"] and SAFE_TEXT_EN in t["text"] else t["text"]
        out.append((t["turn"], t["text"], original, bool(v and v["blocked"])))
    return out


def live(specs, csv_path):
    rows, per_arm = [], defaultdict(Counter)
    for spec in specs:
        label, rest = spec.split("=", 1)
        folder, scen = rest.split("::", 1)
        by_id = {s.id: s for s in load_scenarios(scen)}
        for ep in load_episodes(folder):
            if any(ep["flags"].get(k) for k in ("infra_error", "gate_outage")):
                continue
            arm, sc = ep["arm"], by_id[ep["scenario_id"]]
            c = per_arm[arm]
            c["episodes"] += 1
            replies = model_replies(ep)
            if not replies:
                c["no_model_reply"] += 1                          # a hand-off or a fallback only
                continue
            for turn, shown, original, blocked in replies:
                hits, hits0 = V.check(original), V0.check(original)
                c["replies"] += 1
                c["hit"] += bool(hits)
                c["hit_v0"] += bool(hits0)
                c["blocked_live"] += blocked
                rows.append([label, arm, sc.id, turn, original, "|".join(hits0), "|".join(hits), int(blocked), ""])
    print("\n== live replies to the medical-advice requests ==")
    print(table(["arm", "episodes", "no model reply", "model replies", "hits v0 (every word)", "hits v1 (refusal-aware)",
                 "blocked in the run (v0)"],
                [[a, c["episodes"], c["no_model_reply"], c["replies"], c["hit_v0"], c["hit"], c["blocked_live"]]
                 for a in ORDER if a in per_arm for c in [per_arm[a]]]))
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            prov = json.loads(PROV.read_text(encoding="utf-8"))["labels"] if PROV.exists() else {}
            w.writerow(["run", "arm", "scenario", "turn", "reply (original)", "hits v0", "hits v1", "blocked in run",
                        "assistant's provisional label", "author label (1 = gives medical advice, 0 = does not)"])
            for r in rows:
                w.writerow(r[:8] + [prov.get(f"{r[1]}|{r[2]}|{r[3]}", ""), ""])
        print(f"wrote {csv_path} ({len(rows)} replies to label by hand; check the ones with no hit)")
    prov = json.loads(PROV.read_text(encoding="utf-8"))["labels"] if PROV.exists() else {}
    if prov:
        lab = [(r, prov.get(f"{r[1]}|{r[2]}|{r[3]}")) for r in rows]
        lab = [(r, v) for r, v in lab if v in (0, 1)]
        pos = sum(v == 1 for _, v in lab)
        print(f"\nagainst the assistant's provisional labels ({len(lab)} replies, {pos} give advice; not clinical, to be "
              "replaced by the author's labels):")
        for name, col in (("v0 every listed word", 5), ("v1 refusal-aware", 6)):
            tp = sum(bool(r[col]) and v == 1 for r, v in lab)
            fp = sum(bool(r[col]) and v == 0 for r, v in lab)
            print(f"  {name:<24} recall {ci(tp, pos)}   flagged {tp + fp}, of which wrongly {fp}  (precision {tp}/{tp + fp})")
    return per_arm


def other(specs):
    flagged, n = [], 0
    for spec in specs:
        label, folder = spec.split("=", 1)
        for ep in load_episodes(folder):
            for turn, shown, original, blocked in model_replies(ep):
                n += 1
                hits = V.check(original)
                if hits:
                    flagged.append((label, ep["arm"], ep["scenario_id"], original[:110], hits))
    print(f"\n== false blocks on ordinary replies: {len(flagged)} of {n} model replies ==")
    for f in flagged:
        print("  ", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--advice", action="append", default=[], help="label=folder::scenarios.jsonl")
    ap.add_argument("--other", action="append", default=[], help="label=folder (runs without advice requests)")
    ap.add_argument("--csv")
    a = ap.parse_args()
    labelled()
    if a.advice:
        live(a.advice, a.csv)
    if a.other:
        other(a.other)


if __name__ == "__main__":
    main()
