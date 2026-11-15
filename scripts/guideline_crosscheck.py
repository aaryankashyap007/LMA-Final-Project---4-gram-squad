"""Cross-check the authors' triage rubric (D3 labels) against the WHO/ICRC/MSF Interagency Integrated Triage Tool (IITT).

  python scripts/guideline_crosscheck.py [--out outputs/results/guideline_crosscheck.json] [--csv docs/review/iitt_disagreements.csv]

This is the proposal's fallback for having no clinician ("claim only agreement with a guideline-derived rubric"). It is
NOT clinician review and NOT validation of the labels. What it does:
  - encodes the IITT red (immediate) and yellow (soon) criteria that a patient could describe in a text message, as
    English keyword rules written from the IITT adult and paediatric sheets and the reference card (vital signs, ECG and
    other bedside measurements cannot be seen in a message and are left out);
  - applies them to the English text of the 150 D3 cases, with the case's age (under 12 = paediatric sheet);
  - derives an IITT label per case (red if a red rule fires, else yellow if a yellow rule fires, else none) and compares it
    with ours (EMERGENCY ~ red, URGENT ~ yellow, ROUTINE ~ none; INSUFFICIENT has no IITT counterpart);
  - lists every disagreement for a human to read, and the IITT criteria that no D3 case exercises (coverage gaps).
The keyword rules are crude (a simple negation guard, no understanding), and the IITT is a facility triage tool that also
uses vital signs, so a disagreement is a prompt to look, not an error in either side. Rules were written from the IITT
sheets first and then widened to the wording of D3 messages, which makes the agreement figures optimistic.
Source: WHO, ICRC and MSF, Interagency Integrated Triage Tool, adult and paediatric sheets and reference card
(https://www.who.int/tools/triage), CC BY-NC-SA 3.0 IGO; criteria are paraphrased here, not reproduced.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D3 = ROOT / "data/d3_triage/human_cases/d3_cases.csv"
NEG = re.compile(r"\b(no|not|without|never|nor|none)\b|n't")
GUARD = 30                                   # characters before a match searched for a negation

# (id, tier, who, description, regex, negation guard?, extra test on (age, text) or None)
# who: adult | child | both. Rules are tried on the lower-cased English text.
def _preg(t):
    return bool(re.search(r"pregnan|\d months pregnant", t))


def _two(t):
    classes = [r"drowsy|confused|not waking|very sleepy", r"stiff neck|neck is stiff", r"fever", r"headache"]
    return sum(bool(re.search(c, t)) for c in classes) >= 2


def _dehyd(t):
    return sum(bool(re.search(c, t)) for c in (r"very sleepy|lethargic|limp", r"sunken eyes", r"not drinking|drinks poorly",
                                              r"not passed urine")) >= 2


CRITERIA = [
    ("R01", "red", "both", "unresponsive / collapsed", r"unconscious|unresponsive|not responding|not waking|not moving|collapsed|gone limp|lying on the floor|not breathing", False, None),
    ("R02", "red", "both", "respiratory distress, stridor or blue lips", r"gasping|suffocat|cannot breathe|can't breathe|struggling to breathe|breathing very fast|ribs are pulling|turning blue|full sentence|hard to breathe|choking", False, None),
    ("R03", "red", "both", "heavy bleeding", r"bleeding heavily|bleeding a lot|blood is pouring|blood is flowing heavily|vomiting a lot of blood|soaking quickly|blood is coming from|bleeding heavily with clots", False, None),
    ("R04", "red", "both", "high-risk trauma (hit by vehicle, penetrating, large burn)", r"hit by a (truck|car|bus|vehicle)|stabbed|badly burnt|burnt over a large area", False, None),
    ("R05", "red", "both", "poisoning, ingestion or dangerous exposure", r"swallowed|pesticide|poison|overdose", False, None),
    ("R06", "red", "both", "snake bite", r"snake", False, None),
    ("R07", "red", "adult", "acute chest or abdominal pain, over 50", r"chest pain|chest hurts|on my chest|pain in my chest", True, lambda age, t: age is not None and age > 50),
    ("R08", "red", "adult", "violent or aggressive", r"violent|threatening to hurt|sharp object|hurt all of us", False, None),
    ("R09", "red", "adult", "pregnant with heavy bleeding, severe pain, fit, severe headache, blurred vision, active labour or trauma", r"bleeding heavily|blood is flowing heavily|severe (stomach|abdominal)|terrible stomach pain|fit\b|severe headache|blurry|labou?r pains|head is coming|fell down", True, lambda age, t: _preg(t)),
    ("R10", "red", "both", "active convulsions", r"having a fit|fit right now|shaking and it is not stopping|had a fit and now|just had a fit", False, None),
    ("R11", "red", "both", "any two of altered mental state, stiff neck, fever, headache", r".", False, lambda age, t: _two(t)),
    ("R12", "red", "child", "dehydration signs (two of lethargy, sunken eyes, drinks poorly, no urine)", r".", False, lambda age, t: _dehyd(t)),
    ("R13", "red", "child", "infant under 8 days, or under 2 months with fever", r"\b[0-7] day old|\b[0-1] month old baby (has )?fever|newborn", False, None),
    ("Y01", "yellow", "both", "swelling of face, lips, tongue or throat", r"(face|lips|tongue|throat|mouth).{0,25}swoll?en|swell.{0,25}(face|lips|tongue|throat)", False, None),
    ("Y02", "yellow", "both", "wheezing", r"wheez", True, None),
    ("Y03", "yellow", "both", "vomits everything or ongoing diarrhoea", r"vomiting again and again|keep (water|any food) down|loose motions|diarrh", True, None),
    ("Y04", "yellow", "both", "unable to feed or drink", r"not eating|stopped eating|not drinking|not feeding", True, None),
    ("Y05", "yellow", "both", "ongoing bleeding", r"bleeding", True, lambda age, t: not re.search(r"bleeding (has )?stopped|light bleeding|bleeding is light|light spotting", t)),
    ("Y06", "yellow", "both", "recent fainting", r"faint", True, None),
    ("Y07", "yellow", "both", "other trauma, burn, deformity or fracture", r"\bfell\b|twisted|cut my|metal flew|swollen and painful|cannot move it|burn", True, lambda age, t: "scar" not in t),
    ("Y08", "yellow", "both", "animal bite or time-sensitive exposure", r"dog bit|bit my|stray dog|animal bite", False, None),
    ("Y09", "yellow", "both", "pregnancy with a complication", r"spotting|moving much less|vomiting again|labou?r|swollen|bleeding", True, lambda age, t: _preg(t)),
    ("Y10", "yellow", "both", "weakness, focal neurological or visual complaint", r"weak|slurred|drooping|crooked|cannot lift|cannot find the right words|blurry|vision|spinning", True, None),
    ("Y11", "yellow", "both", "altered mental state or agitation", r"drowsy|confused|restless|irritable|lethargic|hearing voices|seeing things|stopped eating and talking", True, None),
    ("Y12", "yellow", "both", "severe pain", r"severe (pain|headache|stomach pain)|bad pain|terrible .{0,15}pain", True, None),
    ("Y13", "yellow", "both", "unable to pass urine", r"cannot pass urine|not passed urine", True, None),
    ("Y14", "yellow", "child", "infant 8 days to 6 months old", r"\b([1-6]) month old baby|3 months old|7 month old", False, None),
]
# The authors' (non-clinical) reading of each disagreement, so the CSV is reproducible. Groups:
MH = "mental-health crisis: the IITT is a physical-emergency tool and has no self-harm or psychosis criterion (scope difference)"
NEURO = "stroke signs: the IITT puts an acute focal neurological complaint in yellow; our rubric treats it as time-critical"
CONS = "our rubric is more cautious than the IITT category (tie rule: choose the more urgent level)"
INVIS = "the cue is a symptom, history or vital sign the keyword rules or the IITT tiers do not capture"
NOISE = "rule false positive: a keyword fired on text that does not describe the current problem"
SCOPE = "age-based or screening rule of the IITT that does not fit this routine visit"
NOTES = {"C004": CONS, "C005": NEURO, "C017": CONS, "C020": MH, "C022": CONS, "C024": NEURO, "C033": INVIS, "C035": NOISE,
         "C036": NOISE, "C039": INVIS, "C045": MH, "C049": NEURO, "C052": INVIS, "C061": INVIS, "C063": NOISE,
         "C067": CONS, "C068": INVIS, "C078": CONS, "C081": INVIS, "C084": INVIS, "C086": "IITT lists wheezing as yellow; a mild night wheeze in a known asthmatic asking for a check-up is routine in our rubric",
         "C088": INVIS, "C089": MH, "C097": INVIS, "C102": MH, "C107": CONS, "C111": MH, "C119": MH, "C120": CONS,
         "C122": INVIS, "C123": NEURO, "C128": INVIS, "C132": INVIS, "C144": MH, "C149": SCOPE,
         "C053": "POSSIBLE UNDER-TRIAGE: the IITT puts a pregnant patient in active labour in red; ours is URGENT",
         "C127": "POSSIBLE UNDER-TRIAGE: the IITT up-triages a high-risk exposure to red; ours is URGENT (no symptoms, washed off)"}
APPLIES = {"both": lambda child: True, "adult": lambda child: not child, "child": lambda child: child}


def load():
    rows = list(csv.DictReader(open(D3, encoding="utf-8-sig")))
    for r in rows:
        r["age_n"] = int(r["age"]) if r["age"].strip().isdigit() else None
    return rows


def fired(text, age):
    t = text.lower()
    child = age is not None and age < 12
    out = []
    for cid, tier, who, desc, rx, guard, extra in CRITERIA:
        if not APPLIES[who](child):
            continue
        if extra is not None and not extra(age, t):
            continue
        for m in re.finditer(rx, t):
            if guard and NEG.search(t[max(0, m.start() - GUARD):m.start()]):
                continue
            out.append(cid)
            break
    return out


def derive(ids):
    tiers = {next(c[1] for c in CRITERIA if c[0] == i) for i in ids}
    return "red" if "red" in tiers else "yellow" if "yellow" in tiers else "none"


EXPECT = {"EMERGENCY": "red", "URGENT": "yellow", "ROUTINE": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = load()
    for r in rows:
        r["ids"] = fired(r["en"], r["age_n"])
        r["iitt"] = derive(r["ids"])

    print(f"{len(rows)} D3 cases; IITT rules: {sum(c[1] == 'red' for c in CRITERIA)} red, "
          f"{sum(c[1] == 'yellow' for c in CRITERIA)} yellow\n")
    print("== ours (rows) against the IITT-derived tier (columns) ==")
    tiers = ["red", "yellow", "none"]
    conf = Counter((r["label"], r["iitt"]) for r in rows)
    print("label".ljust(14) + "".join(t.ljust(8) for t in tiers))
    for lab in ("EMERGENCY", "URGENT", "ROUTINE", "INSUFFICIENT"):
        print(lab.ljust(14) + "".join(str(conf[(lab, t)]).ljust(8) for t in tiers))

    scored = [r for r in rows if r["label"] in EXPECT]
    exact = sum(EXPECT[r["label"]] == r["iitt"] for r in scored)
    em_red = sum((r["label"] == "EMERGENCY") == (r["iitt"] == "red") for r in scored)
    print(f"\nexact agreement of tier (EMERGENCY=red, URGENT=yellow, ROUTINE=none), INSUFFICIENT excluded: "
          f"{exact}/{len(scored)} = {exact / len(scored):.2f}")
    print(f"emergency-versus-not agreement (our EMERGENCY against IITT red): {em_red}/{len(scored)} = {em_red / len(scored):.2f}")
    under = [r for r in scored if r["iitt"] == "red" and r["label"] != "EMERGENCY"]
    over = [r for r in scored if r["label"] == "EMERGENCY" and r["iitt"] != "red"]
    print(f"IITT red but our label below EMERGENCY (possible under-triage in our rubric): {len(under)}")
    print(f"our EMERGENCY with no IITT red rule firing (rubric stricter, or the rules cannot see the cue): {len(over)}")
    print(f"INSUFFICIENT cases with a red rule firing: {sum(r['label'] == 'INSUFFICIENT' and r['iitt'] == 'red' for r in rows)}")

    cover = defaultdict(Counter)
    for r in rows:
        for i in r["ids"]:
            cover[i][r["label"]] += 1
    print("\n== coverage: D3 cases that exercise each IITT criterion (counts by our label) ==")
    for cid, tier, who, desc, *_ in CRITERIA:
        c = cover.get(cid, Counter())
        print(f"{cid} {tier:<6} {who:<5} {desc[:70]:<70} {sum(c.values()):>3}  "
              + ", ".join(f"{k[:3]}={v}" for k, v in sorted(c.items())))
    print("not exercised by any case:", [c[0] for c in CRITERIA if not cover.get(c[0])])

    dis = [r for r in rows if r["label"] in EXPECT and EXPECT[r["label"]] != r["iitt"]]
    print(f"\n== disagreements to read ({len(dis)}) ==")
    for r in dis:
        print(f"{r['case_id']} ours={r['label'][:3]} iitt={r['iitt']:<6} {','.join(r['ids']) or '-':<12} {r['en'][:88]}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["case", "age", "our label", "IITT-derived tier", "rules fired", "english text", "reading (authors, non-clinical)"])
            for r in dis:
                w.writerow([r["case_id"], r["age"], r["label"], r["iitt"], " ".join(r["ids"]), r["en"], NOTES.get(r["case_id"], "")])
    if a.out:
        Path(a.out).write_text(json.dumps({
            "n": len(rows), "confusion": {f"{k[0]}|{k[1]}": v for k, v in conf.items()},
            "exact": exact, "scored": len(scored), "emergency_vs_not": em_red,
            "under": [r["case_id"] for r in under], "over": [r["case_id"] for r in over],
            "coverage": {k: dict(v) for k, v in cover.items()},
            "disagreements": [{"case": r["case_id"], "ours": r["label"], "iitt": r["iitt"], "rules": r["ids"]} for r in dis]},
            indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
