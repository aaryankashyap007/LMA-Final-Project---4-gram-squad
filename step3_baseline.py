"""Step 3: evaluation harness + keyword-lexicon baseline.

Usage (repo root):  python step3_baseline.py
Creates: src/swasthya/keyword_baseline.py, src/swasthya/evaluate.py,
         scripts/eval_triage.py, tests/test_evaluate.py
Existing files are skipped.
"""
from pathlib import Path

FILES = {}

FILES["src/swasthya/keyword_baseline.py"] = r'''"""Baseline B0: keyword lexicon (script-agnostic: English + Devanagari + Hinglish terms).
Priority: EMERGENCY > URGENT > ROUTINE > INSUFFICIENT (nothing matched).
Deliberately naive: no negation handling, so "no fever" still matches "fever".
"""
import re
import unicodedata

EMERGENCY = {
    "en": ["cant breathe", "cannot breathe", "can not breathe", "not breathing", "breathless",
           "difficulty breathing", "blue lips", "lips are turning blue", "unconscious",
           "not responding", "unresponsive", "fainted", "seizure", "seizures", "fits",
           "convulsion", "heavy bleeding", "bleeding heavily", "bleeding a lot", "poison",
           "poisoning", "overdose", "swallowed", "chest pain", "sweating", "face drooping",
           "slurred", "paralysis", "paralyzed", "end my life", "kill myself", "suicide",
           "not feeding", "limp", "stopped breathing", "choking", "snake bite", "snakebite"],
    "deva": ["सांस नहीं", "सांस लेने में तकलीफ", "सांस फूल", "होंठ नीले", "बेहोश", "जवाब नहीं दे",
             "दौरा", "झटके", "ज़्यादा ब्लीडिंग", "बहुत ब्लीडिंग", "बहुत खून", "खून बंद नहीं",
             "ज़हर", "ओवरडोज़", "सीने में दर्द", "छाती में दर्द", "पसीना", "चेहरा टेढ़ा",
             "बोल नहीं", "लकवा", "जान देना", "आत्महत्या", "मरना चाहता", "दूध नहीं पी",
             "ढीली पड़", "ढीला पड़", "सांप", "दम घुट"],
    "hinglish": ["saans nahi", "sans nahi", "saans lene mein", "saans lene me", "saans phool",
                 "honth neele", "hont neele", "behosh", "jawab nahi", "dora", "daura", "jhatke",
                 "bahut bleeding", "zyada bleeding", "bahut khoon", "khoon band nahi", "zeher",
                 "zahar", "overdose", "seene mein dard", "seene me dard", "seene mein bahut",
                 "chhati mein dard", "chhati me dard", "paseena", "pasina", "chehra tedha",
                 "bol nahi pa", "lakwa", "jaan dena", "jaan de dunga", "khudkushi", "marna chahta",
                 "doodh nahi pi", "dheeli pad", "dheela pad", "saanp", "sanp", "dam ghut"],
}
URGENT = {
    "en": ["fever", "vomit", "vomiting", "deep cut", "cut", "fracture", "broken", "swollen",
           "swelling", "burn", "burns", "diarrhoea", "diarrhea", "severe pain", "bite",
           "stitches", "infection", "pregnant", "pregnancy"],
    "deva": ["बुखार", "उल्टी", "गहरा कट", "कट गया", "फ्रैक्चर", "हड्डी टूट", "सूजन", "जल गया",
             "दस्त", "तेज़ दर्द", "गर्भवती"],
    "hinglish": ["bukhar", "bukhaar", "ulti", "gehra kat", "gehra cut", "chot", "fracture",
                 "haddi toot", "sujan", "jal gaya", "dast", "loose motion", "tez dard",
                 "kat gaya", "pregnant", "garbhwati"],
}
ROUTINE = {
    "en": ["mild", "cough", "cold", "checkup", "check up", "follow up", "followup", "refill",
           "report", "reports", "appointment", "routine", "sugar test", "bp check",
           "back pain", "headache"],
    "deva": ["हल्की", "खांसी", "जुकाम", "चेकअप", "रिपोर्ट", "फॉलो", "अपॉइंटमेंट", "अपॉइन्टमेंट",
             "पुरानी", "दवाई खत्म"],
    "hinglish": ["halki", "khansi", "khasi", "zukam", "sardi", "checkup", "check up", "report",
                 "reports", "follow up", "followup", "appointment", "appoinment", "purana",
                 "dawai khatam", "refill"],
}


def norm(text):
    t = unicodedata.normalize("NFC", text).lower()
    t = t.replace("़", "")          # drop nukta so ज़ == ज
    t = t.replace("ँ", "ं")    # chandrabindu -> anusvara
    t = t.replace("'", "").replace("’", "")   # can't -> cant
    t = re.sub(r"[.,!?;:\"()\[\]।\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _compile(lex):
    terms = sorted({norm(t) for ts in lex.values() for t in ts}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(t) for t in terms) + r")(?![a-z0-9])")


_E, _U, _R = _compile(EMERGENCY), _compile(URGENT), _compile(ROUTINE)


def predict(text):
    t = norm(text)
    if _E.search(t):
        return "EMERGENCY"
    if _U.search(t):
        return "URGENT"
    if _R.search(t):
        return "ROUTINE"
    return "INSUFFICIENT"
'''

FILES["src/swasthya/evaluate.py"] = r'''"""Evaluation harness shared by every triage system (keyword, MuRIL, prompted LLM, QLoRA gate).

Gold file: jsonl from d3.export_jsonl (id, case_id, script, text, label, pair_id, ...).
Preds file: jsonl, one line per gold id: {"id": "...", "pred": "EMERGENCY", "probs": [p_E, p_U, p_R, p_I]}
(probs optional; ordered as schema.LABELS).
"""
import json
from collections import defaultdict

from .schema import LABELS, SCRIPTS
from .metrics import (emergency_recall, over_triage_rate, under_triage_rate, macro_f1, ece,
                      bootstrap_ci, cross_script_recall_gap)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _block(y, p, groups, probs, n_boot):
    r = {"n": len(y), "n_emergency": sum(t == "EMERGENCY" for t in y),
         "emergency_recall": emergency_recall(y, p),
         "emergency_recall_ci": bootstrap_ci(emergency_recall, y, p, groups, n_boot),
         "over_triage": over_triage_rate(y, p), "under_triage": under_triage_rate(y, p),
         "macro_f1": macro_f1(y, p), "macro_f1_ci": bootstrap_ci(macro_f1, y, p, groups, n_boot)}
    if probs is not None:
        r["ece"] = ece(probs, y)
    return r


def pair_stats(gold, preds):
    """Minimal pairs: gold labels differ, so a good system's predictions must differ too."""
    by = defaultdict(list)
    for g in gold:
        if g.get("pair_id"):
            by[(g["pair_id"], g["script"])].append(g)
    out = {s: {"n_pairs": 0, "flip_acc": 0, "both_correct": 0} for s in SCRIPTS}
    for (pid, s), members in by.items():
        if len(members) != 2 or members[0]["label"] == members[1]["label"]:
            continue
        a, b = members
        pa, pb = preds[a["id"]]["pred"], preds[b["id"]]["pred"]
        out[s]["n_pairs"] += 1
        out[s]["flip_acc"] += int(pa != pb)
        out[s]["both_correct"] += int(pa == a["label"] and pb == b["label"])
    for s, d in out.items():
        n = d["n_pairs"]
        d["flip_acc"] = d["flip_acc"] / n if n else None
        d["both_correct"] = d["both_correct"] / n if n else None
    return out


def evaluate(gold, preds, n_boot=1000):
    missing = [g["id"] for g in gold if g["id"] not in preds]
    if missing:
        raise ValueError(f"{len(missing)} gold ids have no prediction, e.g. {missing[:3]}")
    bad = {preds[g["id"]]["pred"] for g in gold} - set(LABELS)
    if bad:
        raise ValueError(f"unknown predicted labels: {bad}")
    y = [g["label"] for g in gold]
    p = [preds[g["id"]]["pred"] for g in gold]
    groups = [g["case_id"] for g in gold]
    scripts = [g["script"] for g in gold]
    has_probs = all("probs" in preds[g["id"]] for g in gold)
    probs = [preds[g["id"]]["probs"] for g in gold] if has_probs else None
    res = {"overall": _block(y, p, groups, probs, n_boot), "by_script": {}}
    for s in SCRIPTS:
        idx = [i for i, x in enumerate(scripts) if x == s]
        if not idx:
            continue
        res["by_script"][s] = _block([y[i] for i in idx], [p[i] for i in idx],
                                     [groups[i] for i in idx],
                                     [probs[i] for i in idx] if probs else None, n_boot)
    gap, recs = cross_script_recall_gap(y, p, scripts)
    res["cross_script_recall_gap"] = gap
    res["emergency_recall_by_script"] = recs
    res["minimal_pairs"] = pair_stats(gold, preds)
    return res
'''

FILES["scripts/eval_triage.py"] = r'''"""Score a triage system on D3 (dev or test).

Keyword baseline:
  python scripts/eval_triage.py --data data/d3_triage/rendered/d3_test.jsonl --keyword --name b0_keyword_test
Any other system (preds.jsonl from Kaggle):
  python scripts/eval_triage.py --data data/d3_triage/rendered/d3_test.jsonl \
      --preds outputs/raw/muril_s42/preds_test.jsonl --name b1_muril_s42_test
Writes outputs/results/<name>.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import evaluate as ev  # noqa: E402
from swasthya.keyword_baseline import predict  # noqa: E402


def f(x):
    return "  n/a" if x is None or x != x else f"{x:.3f}"


def row(name, b):
    lo, hi = b["emergency_recall_ci"]
    return (f"{name:<9} n={b['n']:<4} nE={b['n_emergency']:<3} EmRecall {f(b['emergency_recall'])} "
            f"[{f(lo)},{f(hi)}]  over {f(b['over_triage'])}  under {f(b['under_triage'])}  "
            f"macroF1 {f(b['macro_f1'])}  ECE {f(b.get('ece'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds")
    ap.add_argument("--keyword", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    gold = ev.load_jsonl(a.data)
    if a.keyword:
        preds = {g["id"]: {"id": g["id"], "pred": predict(g["text"])} for g in gold}
    elif a.preds:
        preds = {r["id"]: r for r in ev.load_jsonl(a.preds)}
    else:
        ap.error("give --preds or --keyword")
    res = ev.evaluate(gold, preds, n_boot=a.boot)
    out = ROOT / "outputs/results" / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(row("overall", res["overall"]))
    for s, b in res["by_script"].items():
        print(row(s, b))
    print(f"cross-script emergency-recall gap: {f(res['cross_script_recall_gap'])}")
    for s, d in res["minimal_pairs"].items():
        print(f"pairs[{s}]: n={d['n_pairs']} flip_acc={f(d['flip_acc'])} both_correct={f(d['both_correct'])}")
    print("saved", out)


if __name__ == "__main__":
    main()
'''

FILES["tests/test_evaluate.py"] = r'''import pytest
from swasthya.keyword_baseline import predict
from swasthya import evaluate as ev


def test_keyword_basic():
    assert predict("I cannot breathe") == "EMERGENCY"
    assert predict("मेरे पिताजी को सीने में दर्द है") == "EMERGENCY"
    assert predict("mere pitaji ko seene mein dard hai") == "EMERGENCY"
    assert predict("my son has fever for 3 days") == "URGENT"
    assert predict("I have a mild cough") == "ROUTINE"
    assert predict("meri tabiyat theek nahi hai") == "INSUFFICIENT"
    assert predict("the benefits are fine") == "INSUFFICIENT"   # 'fits' must not match inside 'benefits'


def _gold():
    rows = []
    for cid, lab, pair in [("C1", "EMERGENCY", "P01"), ("C2", "ROUTINE", "P01"), ("C3", "EMERGENCY", None)]:
        for s in ("en", "deva", "hinglish"):
            rows.append({"id": f"{cid}|{s}", "case_id": cid, "script": s, "text": "x",
                         "label": lab, "pair_id": pair})
    return rows


def test_evaluate_perfect_and_pairs():
    gold = _gold()
    preds = {g["id"]: {"pred": g["label"]} for g in gold}
    r = ev.evaluate(gold, preds, n_boot=50)
    assert r["overall"]["emergency_recall"] == 1.0
    assert r["minimal_pairs"]["en"]["flip_acc"] == 1.0
    assert r["cross_script_recall_gap"] == 0.0


def test_evaluate_all_emergency_fails_flip():
    gold = _gold()
    preds = {g["id"]: {"pred": "EMERGENCY"} for g in gold}
    r = ev.evaluate(gold, preds, n_boot=50)
    assert r["overall"]["emergency_recall"] == 1.0
    assert r["overall"]["over_triage"] == 1.0
    assert r["minimal_pairs"]["deva"]["flip_acc"] == 0.0


def test_missing_prediction_raises():
    gold = _gold()
    with pytest.raises(ValueError):
        ev.evaluate(gold, {}, n_boot=10)
'''


def main():
    root = Path(".")
    for path, content in FILES.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            print("skip     ", path)
        else:
            p.write_text(content, encoding="utf-8")
            print("written  ", path)
    print("\nNext: PYTHONPATH=src pytest -q")


if __name__ == "__main__":
    main()
