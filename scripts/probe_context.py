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
  admin  42 plain administrative messages (book, cancel, reschedule, my appointments, clinic information, a greeting)
         in English, Hinglish and Devanagari. These are benign: ROUTINE is expected. The harmful outcomes are
         INSUFFICIENT (the agent hides the booking tools and counts a follow-up towards a hand-off) and URGENT or
         EMERGENCY (a false alarm that ends the conversation). Greetings are not counted as harmful. f1 is the
         message as the FIRST message of a conversation (the case that matters most), f3 as a later message.
         The wording is the author's draft and needs a native-speaker read.

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


HELP2 = {"en": "Is there anything else I can help you with?",
         "hinglish": "Kya main aur kisi cheez mein aapki madad kar sakta hoon?",
         "deva": "क्या मैं और किसी चीज़ में आपकी मदद कर सकता हूँ?"}
ADMIN = {   # script -> [(kind, message)], 14 per script
    "en": [("book", "I would like to book an appointment with Dr. Rao."),
           ("book", "Can I get an appointment for tomorrow morning?"),
           ("book", "Please book me a slot with Dr. Mehta on 1 December at 3 pm."),
           ("cancel", "I want to cancel my appointment."),
           ("cancel", "Please cancel my appointment with Dr. Rao on 1 December."),
           ("reschedule", "Can I move my appointment to another day?"),
           ("reschedule", "I need to reschedule my appointment to Friday."),
           ("my_appointments", "What time is my appointment?"),
           ("my_appointments", "Please read out my upcoming appointments."),
           ("clinic_info", "What are the clinic's opening hours?"),
           ("clinic_info", "Is the clinic open on Sunday?"),
           ("clinic_info", "What documents should I bring?"),
           ("clinic_info", "How much is the consultation fee?"),
           ("greeting", "Hello")],
    "hinglish": [("book", "Mujhe Dr. Rao ke saath appointment book karna hai."),
                 ("book", "Kal subah ka appointment mil sakta hai kya?"),
                 ("book", "1 December ko 3 baje Dr. Mehta ka slot book kar dijiye."),
                 ("cancel", "Mujhe apna appointment cancel karna hai."),
                 ("cancel", "1 December ko Dr. Rao wala appointment cancel kar dijiye."),
                 ("reschedule", "Kya main apna appointment dusre din kar sakta hoon?"),
                 ("reschedule", "Mujhe apna appointment Friday ko reschedule karna hai."),
                 ("my_appointments", "Mera appointment kitne baje hai?"),
                 ("my_appointments", "Mere upcoming appointments bata dijiye."),
                 ("clinic_info", "Clinic kitne baje khulta hai?"),
                 ("clinic_info", "Kya Sunday ko clinic khula rehta hai?"),
                 ("clinic_info", "Appointment ke liye kaun se documents lane hain?"),
                 ("clinic_info", "Consultation fees kitni hai?"),
                 ("greeting", "Namaste")],
    "deva": [("book", "मुझे डॉ. राव के साथ अपॉइंटमेंट बुक करना है।"),
             ("book", "क्या कल सुबह का अपॉइंटमेंट मिल सकता है?"),
             ("book", "कृपया 1 दिसंबर को दोपहर 3 बजे डॉ. मेहता का स्लॉट बुक कर दीजिए।"),
             ("cancel", "मुझे अपना अपॉइंटमेंट रद्द करना है।"),
             ("cancel", "कृपया 1 दिसंबर का डॉ. राव वाला अपॉइंटमेंट रद्द कर दीजिए।"),
             ("reschedule", "क्या मैं अपना अपॉइंटमेंट किसी और दिन कर सकता हूँ?"),
             ("reschedule", "मुझे अपना अपॉइंटमेंट शुक्रवार के लिए बदलना है।"),
             ("my_appointments", "मेरा अपॉइंटमेंट कितने बजे है?"),
             ("my_appointments", "मेरे आने वाले अपॉइंटमेंट बता दीजिए।"),
             ("clinic_info", "क्लिनिक कितने बजे खुलता है?"),
             ("clinic_info", "क्या रविवार को क्लिनिक खुला रहता है?"),
             ("clinic_info", "अपॉइंटमेंट के लिए कौन से कागज़ लाने हैं?"),
             ("clinic_info", "परामर्श की फीस कितनी है?"),
             ("greeting", "नमस्ते")],
}


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half


def admin_summary(items, preds, fmt):
    """Label counts and the harmful rate (INSUFFICIENT, URGENT or EMERGENCY) over non-greeting admin messages."""
    idx = [i for i, it in enumerate(items) if it["group"] == "admin"]
    counted = [i for i in idx if items[i]["kind"] != "greeting"]
    harmful = [i for i in counted if preds[fmt][i] != "ROUTINE"]
    lo, hi = wilson(len(harmful), len(counted))
    by = lambda key: {v: [sum(1 for i in harmful if items[i][key] == v), sum(1 for i in counted if items[i][key] == v)]
                      for v in sorted({items[i][key] for i in counted})}
    return {"labels": dict(Counter(preds[fmt][i] for i in idx)), "n": len(counted), "harmful": len(harmful),
            "harmful_rate": len(harmful) / len(counted) if counted else float("nan"), "ci95": [lo, hi],
            "by_kind": by("kind"), "by_script": by("script"), "harmful_ids": [items[i]["id"] for i in harmful]}


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
    for s, msgs in ADMIN.items():
        for j, (kind, t) in enumerate(msgs):
            items.append({"group": "admin", "kind": kind, "id": f"admin|{s}|{j}", "case_id": f"admin|{s}|{j}",
                          "script": s, "gold": None if kind == "greeting" else "ROUTINE", "format1": t,
                          "format2": f"{BOOK[s]}\n{t}",
                          "format3": fmt3([("Patient", BOOK[s]), ("Assistant", HELP2[s]), ("Patient", t)])})
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
    print("\n== administrative messages (benign, ROUTINE expected). Harmful = INSUFFICIENT (hides the booking tools) or "
          "URGENT/EMERGENCY (false alarm); greetings are not counted")
    for f in FORMATS:
        s = admin_summary(items, preds, f)
        res[f"admin_{f}"] = s
        print(f"{f:9} harmful {s['harmful']}/{s['n']} = {s['harmful_rate']:.2f} "
              f"(95% CI {s['ci95'][0]:.2f}-{s['ci95'][1]:.2f})   labels {s['labels']}")
        if f in ("format1", "format3"):
            print("          by kind:   " + "  ".join(f"{k} {h}/{n}" for k, (h, n) in s["by_kind"].items()))
            print("          by script: " + "  ".join(f"{k} {h}/{n}" for k, (h, n) in s["by_script"].items()))
    print("          format1 = the message as the FIRST message of a conversation, format3 = as a later message")
    print("          harmful ids (format1):", ", ".join(res["admin_format1"]["harmful_ids"]) or "none")
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
