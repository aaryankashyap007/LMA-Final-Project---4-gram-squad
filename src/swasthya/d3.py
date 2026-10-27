"""D3 tooling: load human-written CSVs, validate, parity check, split, export."""
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .schema import LABELS, CATEGORIES, SCRIPTS

REQUIRED = ["case_id", "category", "label", "age", "sex", "cues", "en", "deva",
            "hinglish", "pair_id", "notes"]
DEVA = re.compile(r"[\u0900-\u097F]")


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
            handles[r["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[(r["split"], r["label"])] += 1
    for h in handles.values():
        h.close()
    return dict(counts)
