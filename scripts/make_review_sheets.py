"""Build the two review sheets (native speaker, clinician) and score the clinician sheet once it is filled in.

  python scripts/make_review_sheets.py            # writes docs/review/*
  python scripts/make_review_sheets.py --score docs/review/review_clinician_filled.csv

Outputs (docs/review/):
  review_native.csv             scenario messages (all sets), planted notes, a D3 sample and the fixed Hindi replies,
                                with English, Hinglish and Devanagari side by side and blank reviewer columns
  review_clinician.csv          the rubric, a BLIND label sample of D3 cases (our label is not shown), the emergency
                                and urgent replies and open questions
  _answer_key_clinician.csv     our labels for the sampled cases (keep this away from the clinician)
  REVIEW_MESSAGES.md            instructions and a short message for each reviewer

All data is synthetic. Sampling is seeded, so the sheets are reproducible.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya.orchestrator import FALLBACK, RUBRIC, TEMPLATES  # noqa: E402

OUT = ROOT / "docs/review"
D3 = ROOT / "data/d3_triage/human_cases/d3_cases.csv"
SETS = [("benign", "scenarios_benign_v0.jsonl"), ("adversarial v0", "scenarios_adversarial_v0.jsonl"),
        ("adversarial v1", "scenarios_adversarial_v1.jsonl"), ("adversarial v2", "scenarios_adversarial_v2.jsonl")]
SEED = 13
LABELS = ["EMERGENCY", "URGENT", "ROUTINE", "INSUFFICIENT"]

# the author's English reading of each fixed reply, so a clinician can judge it without reading Devanagari
GLOSS = {"EMERGENCY": "This may be an emergency. Please go to the nearest hospital immediately or call 112 / 108. "
                      "Our staff are also contacting you right now.",
         "URGENT": "Your problem may need to be seen by a doctor today. Our staff will help you fix a time for today. "
                   "Please wait a moment.",
         "UNCLEAR": "I could not fully understand you. Our staff will contact you. Please wait a moment.",
         "ACK": "Our staff are contacting you. Please wait.",
         "FALLBACK": "Sorry, something went wrong. Our staff will help you."}


def d3_rows():
    return list(csv.DictReader(open(D3, encoding="utf-8-sig")))


def scenario_rows():
    """One row per (set, family, turn) with the three scripts side by side; plus the planted notes of v2."""
    rows, notes = [], []
    for label, fname in SETS:
        data = [json.loads(l) for l in (ROOT / "data/d4_clinic" / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        fam = defaultdict(dict)
        for d in data:
            fam[d["tags"][1]][d["tags"][2]] = d
        for f, by in fam.items():
            for turn in range(len(by["en"]["user_turns"])):
                rows.append(["scenario", f"{label} / {f} / turn {turn + 1}", f,
                             by["en"]["user_turns"][turn], by["hinglish"]["user_turns"][turn],
                             by["deva"]["user_turns"][turn]])
            if label == "adversarial v2" and by["en"].get("clinic_patch"):
                def note(script):
                    text = json.dumps(by[script]["clinic_patch"], ensure_ascii=False)
                    start = text.rfind(" [")
                    if start == -1:                                     # the benign twin: a plain update sentence
                        start = max(text.find(" Update:"), text.find(" अपडेट:")) - 1 if (" Update:" in text or " अपडेट:" in text) else -1
                    return text[start + 2:].rstrip("]}\"").replace("\\\"", "\"") if start != -1 else ""
                notes.append(["planted note", f"adversarial v2 / {f}", f, note("en"), note("hinglish"), note("deva")])
    return rows, notes


def native_sheet(rng):
    header = ["section", "item", "family or category", "english", "hinglish", "devanagari",
              "same meaning in all three? (Y/N)", "hinglish: OK or CHANGE", "hinglish: suggested wording",
              "devanagari: OK or CHANGE", "devanagari: suggested wording", "comment"]
    blank = ["", "", "", "", "", ""]
    rows, notes = scenario_rows()
    d3 = d3_rows()
    by_cat = defaultdict(list)
    for r in d3:
        by_cat[r["category"]].append(r)
    sample = []
    for cat in sorted(by_cat):
        sample += rng.sample(by_cat[cat], 4)                      # 7 categories x 4 = 28 cases
    out = [r + blank for r in rows] + [r + blank for r in notes]
    out += [["D3 case", r["case_id"], r["category"], r["en"], r["hinglish"], r["deva"]] + blank for r in sample]
    for k, text in TEMPLATES.items():
        out.append(["fixed reply (Devanagari only)", k, "", GLOSS[k], "", text] + blank)
    out.append(["fixed reply (Devanagari only)", "FALLBACK", "", GLOSS["FALLBACK"], "", FALLBACK] + blank)
    return header, out


def clinician_sheet(rng):
    header = ["section", "item", "age / sex", "category", "text", "your answer", "comment"]
    d3 = d3_rows()
    pairs = defaultdict(list)
    for r in d3:
        if r["pair_id"]:
            pairs[r["pair_id"]].append(r)
    picked = []
    for pid in rng.sample(sorted(pairs), 5):                      # 5 minimal pairs: borderline by construction
        picked += pairs[pid]
    taken = {r["case_id"] for r in picked}
    quota = {"EMERGENCY": 12, "URGENT": 8, "ROUTINE": 6, "INSUFFICIENT": 4}
    for lab, n in quota.items():
        pool = [r for r in d3 if r["label"] == lab and r["case_id"] not in taken and not r["pair_id"]]
        picked += rng.sample(pool, n)
    rng.shuffle(picked)
    key = [[r["case_id"], r["label"], r["pair_id"]] for r in picked]
    out = []
    for line in RUBRIC.split("\n"):
        out.append(["rubric", line[:2].strip(")"), "", "", line, "", ""])
    out.append(["rubric", "tie rule", "", "", "If torn between two levels, choose the more urgent one.", "", ""])
    for r in picked:
        out.append(["label this message", r["case_id"], f"{r['age']} / {r['sex']}", "", r["en"], "", ""])   # no category: no hint
    for k in ("EMERGENCY", "URGENT", "UNCLEAR"):
        out.append(["fixed reply (English reading)", k, "", "", GLOSS[k], "", ""])
    for q in ("Which serious red flags are missing from the rubric, if any?",
              "Is any wording in the rubric unsafe or misleading for a front desk?",
              "Is 'go to the nearest hospital or call 112 / 108' the right instruction for India?",
              "In a clinic front desk, which error is worse: calling a non-emergency an emergency, or the reverse?"):
        out.append(["question", "", "", "", q, "", ""])
    return header, out, key


def write(path, header, rows, bom=True):
    with open(path, "w", newline="", encoding="utf-8-sig" if bom else "utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


MESSAGES = """# Review messages

All items are synthetic: no real patient is described. The system is a research prototype, not a medical device.

## Native-speaker reviewer (Hindi / Hinglish)

> Hi! I am building a clinic chat assistant that understands English, Hindi and Hinglish, and I need a fluent speaker to
> check the wording. Attached is a spreadsheet (`review_native.csv`, open it in Excel or Google Sheets). Each row is a
> message a patient might send, written in English, Hinglish (Hindi in English letters) and Devanagari Hindi. Please
> tell me: (1) do the three versions mean the same thing, (2) does the Hinglish and the Devanagari sound like something
> a real person would write, and if not, how would you say it. There are about 100 rows; reading a sample of 50-60 is
> fine. It should take about an hour. Rows marked 'planted note' are texts that appear inside the clinic's own documents. The last rows are the Hindi replies the assistant gives in an emergency: please
> check they are clear and polite. Could you send it back by <date>? Thank you!

What to fill in per row: column G (Y/N same meaning), H and J (OK or CHANGE), I and K (your wording if CHANGE), L (comment).

## Clinician reviewer

> Hi! I am building a research prototype of a clinic front-desk chat assistant that sorts messages into emergency,
> urgent, routine or too vague. I would value 45-60 minutes of a clinician's judgement. Attached is `review_clinician.csv`.
> (1) Please read the four triage categories at the top and tell me if they are reasonable. (2) Please give your own label
> for each of the 40 short messages (column F), without looking anything up: EMERGENCY, URGENT, ROUTINE or INSUFFICIENT
> (too vague to judge). If torn, choose the more urgent one. (3) Please read the three fixed replies and tell me if
> they are safe, and answer the four questions at the end. All patients are imaginary. Could you send it back by <date>?
> Thank you!

What to fill in per row: column F (your label, or Y/N for the rubric and reply rows), G (comment).

## After the sheets come back
- Native speaker: apply the CHANGE rows to the scenario generators and the D3 text, regenerate the sets, and re-run what changed.
- Clinician: `python scripts/make_review_sheets.py --score docs/review/review_clinician_filled.csv` prints agreement and kappa
  against our labels, and which cases they disagree on. Report the agreement rate; do not silently relabel D3.
"""


def cohen_kappa(a, b, labels):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(path):
    key = {r[0]: r for r in csv.reader(open(OUT / "_answer_key_clinician.csv", encoding="utf-8-sig"))}
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["section"] == "label this message"]
    ours, theirs, cases = [], [], []
    for r in rows:
        mine = r["your answer"].strip().upper()
        mine = {"A": "EMERGENCY", "B": "URGENT", "C": "ROUTINE", "D": "INSUFFICIENT"}.get(mine, mine)
        if mine not in LABELS:
            continue
        ours.append(key[r["item"]][1])
        theirs.append(mine)
        cases.append(r["item"])
    if not ours:
        sys.exit("no labelled rows found")
    n = len(ours)
    print(f"{n} labelled cases; exact agreement {sum(x == y for x, y in zip(ours, theirs))}/{n} "
          f"= {sum(x == y for x, y in zip(ours, theirs)) / n:.2f}; Cohen's kappa {cohen_kappa(ours, theirs, LABELS):.2f}")
    em = [(x == "EMERGENCY", y == "EMERGENCY") for x, y in zip(ours, theirs)]
    print(f"emergency vs not: agreement {sum(a == b for a, b in em)}/{n}; clinician says emergency where we do not: "
          f"{sum((not a) and b for a, b in em)}; we say emergency where they do not: {sum(a and (not b) for a, b in em)}")
    conf = Counter(zip(ours, theirs))
    print("\nours \\ clinician   " + "  ".join(l[:6].ljust(6) for l in LABELS))
    for o in LABELS:
        print(o.ljust(18) + "  ".join(str(conf[(o, t)]).ljust(6) for t in LABELS))
    print("\ndisagreements:", ", ".join(f"{c} (ours {o[:3]}, theirs {t[:3]})" for c, o, t in zip(cases, ours, theirs) if o != t) or "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", help="a filled-in review_clinician.csv")
    a = ap.parse_args()
    if a.score:
        score(a.score)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    h, rows = native_sheet(random.Random(SEED))
    write(OUT / "review_native.csv", h, rows)
    h2, rows2, key = clinician_sheet(random.Random(SEED))
    write(OUT / "review_clinician.csv", h2, rows2)
    write(OUT / "_answer_key_clinician.csv", ["case_id", "our_label", "pair_id"], key)
    (OUT / "REVIEW_MESSAGES.md").write_text(MESSAGES, encoding="utf-8")
    print(f"native sheet: {len(rows)} rows; clinician sheet: {len(rows2)} rows "
          f"({sum(r[0] == 'label this message' for r in rows2)} cases to label); key: {len(key)} cases -> {OUT.relative_to(ROOT)}")
    print("our label mix in the sample:", dict(Counter(k[1] for k in key)))


if __name__ == "__main__":
    main()
