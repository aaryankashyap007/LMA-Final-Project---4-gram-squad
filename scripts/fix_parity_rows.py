"""Adjudicate the 7 rows flagged by the second-annotator parity check.

Run from repo root:  python scripts/fix_parity_rows.py
- Backs up the original to data/d3_triage/human_cases/archive/d3_cases_before_adjudication.csv
  (kept OUT of the validator's glob, so it is never loaded as data).
- Edits only the 7 rows below, only if the current text matches what was reviewed.
- Sets a2_extra=no on fixed rows; where cues changed, a2_present_* are set to the new cue list.
- Safe to re-run (already-fixed rows are skipped).
"""
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/d3_triage/human_cases/d3_cases.csv"
BACKUP = CSV.parent / "archive" / "d3_cases_before_adjudication.csv"

# case_id -> {"cues": new cues or None, "en"/"deva"/"hinglish": (old, new), "why": note}
FIXES = {
    "C002": {
        "cues": "overdose;relative_daughter;swallowed_whole_strip;very_sleepy",
        "why": "cues were incomplete (whole strip, very sleepy); texts unchanged",
    },
    "C004": {
        "cues": "fall_from_bike;head_hit;relative_friend;very_drowsy;vomiting",
        "why": "'very drowsy' made a cue; texts unchanged",
    },
    "C010": {
        "en": ("I am feeling very anxious and I don't know why.",
               "I am feeling anxious and I don't know why."),
        "deva": ("मुझे बहुत घबराहट हो रही है और पता नहीं क्यों।",
                 "मुझे घबराहट हो रही है और पता नहीं क्यों।"),
        "hinglish": ("mujhe bahut ghabrahat ho rahi hai aur pata nahi kyun",
                     "mujhe ghabrahat ho rahi hai aur pata nahi kyun"),
        "why": "removed unlisted intensifier 'very'",
    },
    "C051": {
        "en": ("I am very depressed and I don't know what to do.",
               "I am depressed and I don't know what to do."),
        "deva": ("मैं बहुत उदास हूँ और मुझे समझ नहीं आ रहा कि क्या करूँ।",
                 "मैं उदास हूँ और मुझे समझ नहीं आ रहा कि क्या करूँ।"),
        "hinglish": ("main bahut udaas hoon aur mujhe samajh nahi aa raha ki kya karun",
                     "main udaas hoon aur mujhe samajh nahi aa raha ki kya karun"),
        "why": "removed unlisted intensifier 'very'",
    },
    "C062": {
        "cues": "duration_2weeks;poor_sleep;stress;wants_counsellor;work_pressure",
        "deva": ("काम के दबाव से मुझे बहुत तनाव है और 2 हफ़्ते से मेरी नींद ठीक नहीं आ रही। मैं किसी काउंसलर से बात करना चाहता हूँ।",
                 "काम के दबाव से मुझे तनाव है और 2 हफ़्ते से मेरी नींद ठीक नहीं आ रही। मैं किसी काउंसलर से बात करना चाहता हूँ।"),
        "hinglish": ("kaam ke pressure se mujhe bahut tension hai aur 2 hafte se meri neend theek nahi aa rahi. main kisi counsellor se baat karna chahta hoon",
                     "kaam ke pressure se mujhe tension hai aur 2 hafte se meri neend theek nahi aa rahi. main kisi counsellor se baat karna chahta hoon"),
        "why": "added cue wants_counsellor; removed 'bahut' from deva/hinglish (en had none)",
    },
    "C090": {
        "en": ("My little girl has some problem since yesterday.",
               "My daughter has some problem since yesterday."),
        "deva": ("मेरी छोटी बेटी को कल से कुछ दिक्कत है।",
                 "मेरी बेटी को कल से कुछ दिक्कत है।"),
        "hinglish": ("meri chhoti beti ko kal se kuch dikkat hai",
                     "meri beti ko kal se kuch dikkat hai"),
        "why": "removed 'little' (implied young age, not in cues)",
    },
    "C115": {
        "en": ("I got my report and I am very worried. What should I do?",
               "I got my report and I am worried. What should I do?"),
        "deva": ("मुझे अपनी रिपोर्ट मिली है और मैं बहुत चिंतित हूँ। मुझे क्या करना चाहिए?",
                 "मुझे अपनी रिपोर्ट मिली है और मैं चिंतित हूँ। मुझे क्या करना चाहिए?"),
        "hinglish": ("mujhe apni report mili hai aur main bahut chintit hoon. mujhe kya karna chahiye",
                     "mujhe apni report mili hai aur main chintit hoon. mujhe kya karna chahiye"),
        "why": "removed unlisted intensifier 'very'",
    },
}


def main():
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    if not BACKUP.exists():
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(CSV, BACKUP)
        print("backup  ->", BACKUP.relative_to(ROOT))

    by_id = {r["case_id"].strip(): r for r in rows}
    changed = 0
    for cid, fx in FIXES.items():
        r = by_id.get(cid)
        if r is None:
            print(f"WARN {cid}: not found, skipped")
            continue
        did = False
        stale = False
        for col in ("en", "deva", "hinglish"):
            if col not in fx:
                continue
            old, new = fx[col]
            cur = r[col].strip()
            if cur == new:
                continue
            if cur != old:
                print(f"WARN {cid}.{col}: text differs from what was reviewed, skipped.\n     current: {cur!r}")
                stale = True
                continue
            r[col] = new
            did = True
        if fx.get("cues") and r["cues"].strip() != fx["cues"]:
            r["cues"] = fx["cues"]
            for s in ("en", "deva", "hinglish"):
                r[f"a2_present_{s}"] = fx["cues"]
            did = True
        if did and not stale:
            r["a2_extra"] = "no"
            note = f"adjudicated: {fx['why']}"
            r["notes"] = (r["notes"] + " | " if r["notes"].strip() else "") + note
            changed += 1
            print(f"fixed   {cid}: {fx['why']}")
        elif not did and not stale:
            print(f"skip    {cid}: already fixed")

    with open(CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{changed} row(s) changed. Next: python scripts/validate_d3.py --export")


if __name__ == "__main__":
    main()
