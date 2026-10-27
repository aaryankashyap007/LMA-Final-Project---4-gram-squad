"""Step 2: D3 human-written evaluation set tooling.

Usage (inside your repo root):  python step2_d3.py
Creates/updates: .gitignore (overwritten), docs/D3_GUIDE.md, src/swasthya/d3.py,
scripts/validate_d3.py, tests/test_d3.py, data/d3_triage/human_cases/template.csv
Other existing files are skipped.
"""
from pathlib import Path

OVERWRITE = {".gitignore"}
FILES = {}

FILES[".gitignore"] = '''# --- python ---
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/

# --- secrets / env ---
.env
*.key
kaggle.json

# --- big model artifacts (never commit) ---
*.safetensors
*.bin
*.pt
*.pth
*.gguf
*.ckpt
adapters/
checkpoints/
wandb/

# --- raw Kaggle downloads (keep only small results) ---
outputs/raw/*
!outputs/raw/.gitkeep

# --- Kaggle upload bundles ---
*.zip
kaggle_bundle/

# --- editor / OS ---
.vscode/
.idea/
.DS_Store
*.swp
*~
'''

FILES["docs/D3_GUIDE.md"] = '''# D3 writing guide (human-written eval set)

Goal: ~100 cases total, 25 per person (initials prefix: AK, HN, PM, SM).
Each case = ONE row in a CSV, with the same clinical content written 3 ways.
Time: ~3 min/case.  Copy `data/d3_triage/human_cases/template.csv` to
`data/d3_triage/human_cases/<INITIALS>.csv` and fill it (LibreOffice/Sheets is fine; save as UTF-8 CSV).

## Columns
| column | what to write |
|---|---|
| case_id | `AK001`, `AK002`, ... (your initials + running number) |
| category | one of: airway_breathing_consciousness, heart_brain, bleeding_injury_poisoning, maternal, newborn_child, general_medical, mental_health_crisis |
| label | EMERGENCY / URGENT / ROUTINE / INSUFFICIENT |
| age | number (or blank) |
| sex | M / F / blank |
| cues | the clinical cues, `;`-separated, lowercase snake_case, e.g. `chest_pain;sweating;pain_left_arm;onset_30min` |
| en | English message, as a patient/relative would type it |
| deva | SAME content in Devanagari Hindi |
| hinglish | SAME content in romanised Hindi, hand-typed the way you would text (spelling variation OK) |
| pair_id | blank, or e.g. `AK-P1` for the two cases of a minimal pair |
| notes | optional |

## THE parity rule (most important)
Write the `cues` list FIRST. Then each of en / deva / hinglish must mention exactly those cues:
nothing missing, nothing extra (no extra symptom, no extra duration, no extra age hint).
Wording, spelling, politeness and filler may differ. Clinical content may not.

## Label rubric (planning version, to be clinician-reviewed)
- EMERGENCY: immediate danger to life -> staff alert now. e.g. cannot breathe / blue lips,
  chest pain with sweating or arm pain, face droop + slurred speech, unconscious/fits,
  heavy bleeding, poisoning, pregnant with heavy bleeding or no fetal movement + severe pain,
  newborn not feeding/limp/blue, suicidal intent with a plan.
- URGENT: needs a clinician the same day, not life threatening this minute. e.g. fever 3 days in a child
  that is drinking, deep cut that has stopped bleeding, painful urination with fever, persistent vomiting.
- ROUTINE: can wait for a normal appointment. e.g. mild cough 1 week, follow-up, BP check, old back pain.
- INSUFFICIENT: message too vague to triage ("tabiyat theek nahi hai", "doctor se milna hai, dikkat hai").

## Quotas per person (25 cases)
- 10 EMERGENCY, 6 URGENT, 5 ROUTINE, 4 INSUFFICIENT
- cover all 7 categories (3-4 each)
- 4 minimal pairs (= 8 of the 25 cases): two cases identical except ONE cue that flips the label.
  Example: `chest_pain;sweating;pain_left_arm` (EMERGENCY) vs `chest_pain;pain_on_pressing_chest_wall;after_gym` (ROUTINE).
  Give both rows the same pair_id.
- Write like real patients: short, typos, no medical jargon, sometimes a relative writing ("meri maa ko ...").
- Mix: some messages 1 line, some 3-4 lines; a few with an appointment request too ("kal ka appointment chahiye").
- All fake. No real names/phone numbers.

## Second-annotator parity check (after writing)
Swap CSVs with a teammate. They read each rendering SEPARATELY and fill:
`a2_present_en`, `a2_present_deva`, `a2_present_hinglish` = which cues from the `cues` column are present in that text (`;`-separated),
`a2_extra` = `yes` if ANY rendering has a clinical detail not in `cues`, else `no`.
Then send the file back. Validator reports the parity-pass rate per script.

## Checking your file
    python scripts/validate_d3.py              # validate all CSVs in human_cases/
    python scripts/validate_d3.py --export     # also write dev/test jsonl splits
'''

FILES["data/d3_triage/human_cases/template.csv"] = (
    "case_id,category,label,age,sex,cues,en,deva,hinglish,pair_id,notes,"
    "a2_present_en,a2_present_deva,a2_present_hinglish,a2_extra\n"
    'EX001,heart_brain,EMERGENCY,58,M,chest_pain;sweating;pain_left_arm,'
    '"My father has chest pain with sweating and his left arm hurts.",'
    '"मेरे पिताजी को सीने में दर्द है, पसीना आ रहा है और बाएं हाथ में दर्द है।",'
    '"mere pitaji ko seene mein dard hai, paseena aa raha hai aur baye haath mein bhi dard hai",'
    'EX-P1,example row - delete,,,,\n'
    'EX002,heart_brain,ROUTINE,25,M,chest_pain;pain_on_pressing_chest_wall;after_gym,'
    '"I have chest pain when I press my chest wall since I lifted weights.",'
    '"जिम में वज़न उठाने के बाद छाती दबाने पर दर्द होता है।",'
    '"gym mein weight uthane ke baad chhati dabane pe dard hota hai",'
    'EX-P1,example row - delete,,,,\n'
)

FILES["src/swasthya/d3.py"] = '''"""D3 tooling: load human-written CSVs, validate, parity check, split, export."""
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .schema import LABELS, CATEGORIES, SCRIPTS

REQUIRED = ["case_id", "category", "label", "age", "sex", "cues", "en", "deva",
            "hinglish", "pair_id", "notes"]
DEVA = re.compile(r"[\\u0900-\\u097F]")


def parse_cues(s):
    return sorted({c.strip().lower() for c in (s or "").split(";") if c.strip()})


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if r.get("case_id") and not r["case_id"].upper().startswith("EX"):
            out.append(r)
    return out


def deva_ratio(text):
    chars = [c for c in text if c.isalnum() or DEVA.match(c)]
    if not chars:
        return 0.0
    return sum(1 for c in chars if DEVA.match(c)) / len(chars)


def validate_rows(rows):
    errors, warnings = [], []
    if rows:
        missing = [c for c in REQUIRED if c not in rows[0]]
        if missing:
            return [f"missing columns: {missing}"], warnings
    seen = set()
    pairs = defaultdict(list)
    for r in rows:
        cid = r["case_id"]
        if cid in seen:
            errors.append(f"{cid}: duplicate case_id")
        seen.add(cid)
        if r["label"] not in LABELS:
            errors.append(f"{cid}: bad label {r['label']!r}")
        if r["category"] not in CATEGORIES:
            errors.append(f"{cid}: bad category {r['category']!r}")
        if r["age"] and not r["age"].isdigit():
            errors.append(f"{cid}: age must be a number")
        if not parse_cues(r["cues"]):
            errors.append(f"{cid}: cues empty")
        for s in SCRIPTS:
            if len(r[s]) < 8:
                errors.append(f"{cid}: {s} text missing/too short")
        if r["deva"] and deva_ratio(r["deva"]) < 0.4:
            errors.append(f"{cid}: deva text is not mostly Devanagari")
        for s in ("en", "hinglish"):
            if DEVA.search(r[s]):
                errors.append(f"{cid}: {s} contains Devanagari characters")
        if r["en"] and r["hinglish"] and r["en"].lower() == r["hinglish"].lower():
            warnings.append(f"{cid}: hinglish identical to English")
        if r["pair_id"]:
            pairs[r["pair_id"]].append(r)
    for pid, members in pairs.items():
        if len(members) != 2:
            errors.append(f"pair {pid}: has {len(members)} members (need 2)")
        elif members[0]["label"] == members[1]["label"]:
            errors.append(f"pair {pid}: both members have label {members[0]['label']} (must differ)")
    return errors, warnings


def parity_report(rows):
    """Second-annotator parity: rendering passes if it contains all gold cues and a2_extra != yes."""
    out = {s: [0, 0] for s in SCRIPTS}  # passes, total
    for r in rows:
        if not r.get("a2_present_en") and not r.get("a2_present_deva") and not r.get("a2_present_hinglish"):
            continue
        gold = set(parse_cues(r["cues"]))
        extra = r.get("a2_extra", "").lower() == "yes"
        for s in SCRIPTS:
            col = {"en": "a2_present_en", "deva": "a2_present_deva", "hinglish": "a2_present_hinglish"}[s]
            present = set(parse_cues(r.get(col, "")))
            out[s][1] += 1
            out[s][0] += int(present == gold and not extra)
    return {s: (p / t if t else None, t) for s, (p, t) in out.items()}


def assign_splits(rows, dev_frac=0.3, seed=13):
    """Group-aware split: both members of a minimal pair stay together. Stratified on has-emergency."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["pair_id"] or r["case_id"]].append(r)
    strata = defaultdict(list)
    for g, members in groups.items():
        key = any(m["label"] == "EMERGENCY" for m in members)
        strata[key].append(g)
    rng = random.Random(seed)
    split_of = {}
    for key in sorted(strata):
        gs = sorted(strata[key])
        rng.shuffle(gs)
        n_dev = round(len(gs) * dev_frac)
        for i, g in enumerate(gs):
            split_of[g] = "dev" if i < n_dev else "test"
    for g, members in groups.items():
        for m in members:
            m["split"] = split_of[g]
    return rows


def export_jsonl(rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {sp: open(out_dir / f"d3_{sp}.jsonl", "w", encoding="utf-8") for sp in ("dev", "test")}
    counts = defaultdict(int)
    for r in rows:
        for s in SCRIPTS:
            rec = {"id": f"{r['case_id']}|{s}", "case_id": r["case_id"], "script": s,
                   "text": r[s], "label": r["label"], "category": r["category"],
                   "pair_id": r["pair_id"] or None, "split": r["split"],
                   "cues": parse_cues(r["cues"])}
            handles[r["split"]].write(json.dumps(rec, ensure_ascii=False) + "\\n")
            counts[(r["split"], r["label"])] += 1
    for h in handles.values():
        h.close()
    return dict(counts)
'''

FILES["scripts/validate_d3.py"] = '''"""Validate human-written D3 CSVs; optionally export dev/test jsonl.
Run from repo root:  python scripts/validate_d3.py [--export]
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import d3  # noqa: E402


def main():
    files = sorted(p for p in (ROOT / "data/d3_triage/human_cases").glob("*.csv") if p.name != "template.csv")
    if not files:
        print("No CSVs found in data/d3_triage/human_cases/ (template.csv is ignored)")
        return 1
    rows = []
    for f in files:
        r = d3.load_csv(f)
        print(f"{f.name}: {len(r)} cases")
        rows += r
    errors, warnings = d3.validate_rows(rows)
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"\\nTotal {len(rows)} cases | labels {dict(Counter(r['label'] for r in rows))}")
    print(f"categories {dict(Counter(r['category'] for r in rows))}")
    rep = d3.parity_report(rows)
    for s, (rate, n) in rep.items():
        print(f"parity[{s}]: {'n/a' if rate is None else f'{rate:.2%}'} (n={n})")
    if errors:
        print(f"\\n{len(errors)} error(s) - fix before exporting.")
        return 1
    if "--export" in sys.argv:
        rows = d3.assign_splits(rows)
        counts = d3.export_jsonl(rows, ROOT / "data/d3_triage/rendered")
        print("exported to data/d3_triage/rendered/  (split,label)->renderings:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

FILES["tests/test_d3.py"] = '''from swasthya import d3


def row(cid, label="ROUTINE", pair="", **kw):
    r = dict(case_id=cid, category="general_medical", label=label, age="30", sex="M",
             cues="cough;one_week", en="I have had a cough for one week.",
             deva="मुझे एक हफ्ते से खांसी है।", hinglish="mujhe ek hafte se khansi hai",
             pair_id=pair, notes="")
    r.update(kw)
    return r


def test_valid_rows_pass():
    errs, _ = d3.validate_rows([row("A1"), row("A2")])
    assert errs == []


def test_bad_rows_flagged():
    errs, _ = d3.validate_rows([row("A1", deva="mujhe khansi hai bahut"), row("A1", label="BAD")])
    joined = " ".join(errs)
    assert "Devanagari" in joined and "duplicate" in joined and "bad label" in joined


def test_pair_must_differ_and_stay_together():
    rows = [row("P1", "EMERGENCY", "X-P1"), row("P2", "ROUTINE", "X-P1")] + \\
           [row(f"C{i}", "EMERGENCY") for i in range(10)]
    assert d3.validate_rows(rows)[0] == []
    d3.assign_splits(rows)
    assert rows[0]["split"] == rows[1]["split"]
    bad = [row("P1", "ROUTINE", "Z"), row("P2", "ROUTINE", "Z")]
    assert any("must differ" in e for e in d3.validate_rows(bad)[0])


def test_parity_report():
    r = row("A1", a2_present_en="cough;one_week", a2_present_deva="cough", a2_present_hinglish="cough;one_week", a2_extra="no")
    rep = d3.parity_report([r])
    assert rep["en"][0] == 1.0 and rep["deva"][0] == 0.0
'''


def main():
    root = Path(".")
    (root / "data/d3_triage/human_cases").mkdir(parents=True, exist_ok=True)
    (root / "outputs/raw").mkdir(parents=True, exist_ok=True)
    (root / "outputs/raw/.gitkeep").touch()
    for path, content in FILES.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and path not in OVERWRITE:
            print("skip     ", path)
        else:
            p.write_text(content, encoding="utf-8")
            print("written  ", path)
    print("\nNext: PYTHONPATH=src pytest -q")


if __name__ == "__main__":
    main()

