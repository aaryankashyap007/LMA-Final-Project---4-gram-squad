"""Does G1 still work when the gate sees dialogue context instead of one standalone message?

G1 was trained on single messages. The agent can feed the gate (1) the last message only, (2) the last two patient
messages, or (3) the same with the assistant's question. This probe measures all three on the D3 dev split plus
a few hand-written dialogue cases.

  build : write probe_items.json (full records) and probe_texts.json (texts per format, aligned with the items)
  score : read probe_items.json + probe_out.json (labels per format, made on a GPU) and print the comparison

Groups
  dev    every dev rendering. f1 = text alone, f2 = a routine booking message + text, f3 = "Patient/Assistant"
         transcript ending in the text. Gold = the case label. Shows whether context hurts emergency recall or adds
         false alarms.
  short  non-clinical replies ("haan ji", "11 am", "thanks"). No gold: the harmful outcome is INSUFFICIENT (hides
         booking tools) and EMERGENCY/URGENT (false alarm).
  follow hand-written follow-up answers where the label depends on context. EXPECTED labels are the author's
         judgement and need a second reader.

Run on a GPU (Kaggle) between the two steps:
  backend = load_backend(adapter_dir=...); out = {f: [decide(p) for p in backend.probs(t)] for f, t in texts.items()}
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swasthya.metrics import bootstrap_ci, emergency_recall, over_triage_rate  # noqa: E402

FORMATS = ["format1", "format2", "format3"]
ASK = {"en": "What problem are you facing?", "deva": "आपको क्या समस्या है?", "hinglish": "Aapko kya problem hai?"}
BOOK = {"en": "I want to book an appointment for my check-up.",
        "deva": "मुझे चेकअप के लिए अपॉइंटमेंट बुक करना है।",
        "hinglish": "Mujhe check-up ke liye appointment book karna hai."}
CONFIRM_Q = {"en": "Shall I book Dr. Rao on 1 December at 3 pm?",
             "deva": "क्या मैं डॉ. राव के साथ 1 दिसंबर को 3 बजे बुक कर दूँ?",
             "hinglish": "Kya main Dr. Rao ke saath 1 December ko 3 baje book kar doon?"}
SHORT = {
    "en": ["yes", "yes please", "ok", "11 am", "tomorrow morning", "thanks", "no problem", "that works"],
    "hinglish": ["haan ji", "haan kar do", "theek hai", "kal subah", "11 baje", "dhanyawad", "ok sir", "koi baat nahi"],
    "deva": ["हाँ जी", "ठीक है", "कल सुबह", "11 बजे", "धन्यवाद", "जी हाँ कर दीजिए", "बिल्कुल", "कोई बात नहीं"],
}
FOLLOW = [   # (script, first message, assistant question, patient answer, expected label)
    ("en", "I have had a cough for a few weeks", "Do you have a fever or trouble breathing?", "no", "ROUTINE"),
    ("en", "My father is not feeling well", "Is he awake and responding?", "no he is not responding", "EMERGENCY"),
    ("en", "I need a repeat prescription", "Any new symptoms?", "none", "ROUTINE"),
    ("hinglish", "Meri beti ko bukhar hai", "Kya usko saans lene mein dikkat hai?", "nahi, bas bukhar hai", "URGENT"),
    ("hinglish", "Mummy ko chakkar aa rahe hain", "Kya woh hosh mein hain?", "nahi, behosh ho gayi", "EMERGENCY"),
    ("hinglish", "Bacche ko ulti ho rahi hai", "Kab se?", "kal raat se", "URGENT"),
    ("deva", "मुझे कई दिनों से हल्की खांसी है", "क्या बुखार या सांस की तकलीफ है?", "नहीं", "ROUTINE"),
    ("deva", "मेरे पति को सीने में दर्द है", "क्या दर्द बढ़ रहा है?", "हाँ और पसीना भी आ रहा है", "EMERGENCY"),
]


def fmt3(turns):
    """turns: list of (speaker, text) -> 'Patient: ...\\nAssistant: ...' transcript."""
    return "\n".join(f"{who}: {t}" for who, t in turns)


def build_items(dev_rows):
    items = []
    routine = defaultdict(list)
    for r in sorted(dev_rows, key=lambda r: r["id"]):
        if r["label"] == "ROUTINE":
            routine[r["script"]].append(r)
    for i, r in enumerate(sorted(dev_rows, key=lambda r: r["id"])):
        pool = [x for x in routine[r["script"]] if x["case_id"] != r["case_id"]]
        pre = pool[i % len(pool)]["text"]
        s = r["script"]
        items.append({"group": "dev", "id": r["id"], "case_id": r["case_id"], "script": s, "gold": r["label"],
                      "format1": r["text"], "format2": f"{pre}\n{r['text']}",
                      "format3": fmt3([("Patient", pre), ("Assistant", ASK[s]), ("Patient", r["text"])])})
    for s, replies in SHORT.items():
        for j, t in enumerate(replies):
            items.append({"group": "short", "id": f"short|{s}|{j}", "case_id": f"short|{s}|{j}", "script": s,
                          "gold": None, "format1": t, "format2": f"{BOOK[s]}\n{t}",
                          "format3": fmt3([("Patient", BOOK[s]), ("Assistant", CONFIRM_Q[s]), ("Patient", t)])})
    for j, (s, first, q, a, exp) in enumerate(FOLLOW):
        items.append({"group": "follow", "id": f"follow|{s}|{j}", "case_id": f"follow|{s}|{j}", "script": s,
                      "gold": exp, "format1": a, "format2": f"{first}\n{a}",
                      "format3": fmt3([("Patient", first), ("Assistant", q), ("Patient", a)])})
    return items


def cmd_build(a):
    rows = [json.loads(l) for l in open(a.dev, encoding="utf-8") if l.strip()]
    items = build_items(rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "probe_texts.json").write_text(
        json.dumps({f: [it[f] for it in items] for f in FORMATS}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} items ({Counter(i['group'] for i in items)}) -> {out}")


def _ci(fn, gold, pred, groups, n):
    lo, hi = bootstrap_ci(fn, gold, pred, groups, n)
    return f"[{lo:.2f},{hi:.2f}]"


def cmd_score(a):
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))
    preds = json.loads(Path(a.preds).read_text(encoding="utf-8"))
    for f in FORMATS:
        if len(preds.get(f, [])) != len(items):
            raise SystemExit(f"{f}: {len(preds.get(f, []))} predictions for {len(items)} items")
    res = {}
    dev = [i for i, it in enumerate(items) if it["group"] == "dev"]
    print("== dev split: does context change triage? (95% CI bootstrapped over cases)")
    print(f"{'format':9} {'emergency recall':>24} {'over-triage':>22}")
    for f in FORMATS:
        g = [items[i]["gold"] for i in dev]
        p = [preds[f][i] for i in dev]
        c = [items[i]["case_id"] for i in dev]
        rec, ot = emergency_recall(g, p), over_triage_rate(g, p)
        res[f"dev_{f}"] = {"emergency_recall": rec, "over_triage": ot}
        print(f"{f:9} {rec:8.3f} {_ci(emergency_recall, g, p, c, a.boot):>15} "
              f"{ot:8.3f} {_ci(over_triage_rate, g, p, c, a.boot):>13}")
    print("\n== short non-clinical replies: label shares (harmful: INSUFFICIENT hides booking tools)")
    sh = [i for i, it in enumerate(items) if it["group"] == "short"]
    for f in FORMATS:
        cnt = Counter(preds[f][i] for i in sh)
        res[f"short_{f}"] = {k: cnt.get(k, 0) / len(sh) for k in ("EMERGENCY", "URGENT", "ROUTINE", "INSUFFICIENT")}
        print(f"{f:9} " + "  ".join(f"{k}={cnt.get(k, 0)}/{len(sh)}" for k in
                                    ("EMERGENCY", "URGENT", "ROUTINE", "INSUFFICIENT")))
    print("\n== follow-up cases (expected labels are the author's judgement): accuracy and misses")
    fo = [i for i, it in enumerate(items) if it["group"] == "follow"]
    for f in FORMATS:
        ok = sum(preds[f][i] == items[i]["gold"] for i in fo)
        miss = [f"{items[i]['id']}:{items[i]['gold']}->{preds[f][i]}" for i in fo if preds[f][i] != items[i]["gold"]]
        res[f"follow_{f}"] = {"accuracy": ok / len(fo)}
        print(f"{f:9} {ok}/{len(fo)} correct" + (f"   misses: {', '.join(miss)}" if miss else ""))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nsaved {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dev", default="data/d3_triage/rendered/d3_dev.jsonl")
    b.add_argument("--out_dir", default="outputs/probe")
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("score")
    s.add_argument("--items", default="outputs/probe/probe_items.json")
    s.add_argument("--preds", default="outputs/probe/probe_out.json")
    s.add_argument("--out", default="outputs/probe/probe_results.json")
    s.add_argument("--boot", type=int, default=1000)
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
