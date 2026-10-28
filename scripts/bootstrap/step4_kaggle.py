"""Step 4: training data prep, Kaggle scripts (MuRIL / prompted LLM / QLoRA gate), bundling, scoring.

Usage (repo root):  python step4_kaggle.py
Overwrites scripts/validate_d3.py (patched: only reads d3_cases.csv). Other existing files are skipped.
"""
from pathlib import Path

OVERWRITE = {"scripts/validate_d3.py"}
FILES = {}

# ----------------------------------------------------------------------------------------------
FILES["scripts/validate_d3.py"] = r'''"""Validate the human-written D3 EVAL set (d3_cases.csv); optionally export dev/test jsonl.
Run from repo root:  python scripts/validate_d3.py [--export]
(Training data is handled by scripts/prepare_train.py.)
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import d3  # noqa: E402


def main():
    f = ROOT / "data/d3_triage/human_cases/d3_cases.csv"
    if not f.exists():
        print("Missing", f)
        return 1
    rows = d3.load_csv(f)
    print(f"{f.name}: {len(rows)} cases")
    errors, warnings = d3.validate_rows(rows)
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"\nTotal {len(rows)} cases | labels {dict(Counter(r['label'] for r in rows))}")
    print(f"categories {dict(Counter(r['category'] for r in rows))}")
    rep = d3.parity_report(rows)
    for s, (rate, n) in rep.items():
        print(f"parity[{s}]: {'n/a' if rate is None else f'{rate:.2%}'} (n={n})")
    if errors:
        print(f"\n{len(errors)} error(s) - fix before exporting.")
        return 1
    if "--export" in sys.argv:
        rows = d3.assign_splits(rows)
        counts = d3.export_jsonl(rows, ROOT / "data/d3_triage/rendered")
        print("exported to data/d3_triage/rendered/  (split,label)->renderings:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# ----------------------------------------------------------------------------------------------
FILES["src/swasthya/kio.py"] = r'''"""Small IO helpers shared by Kaggle scripts and local scoring (no torch dependency)."""
import json
from pathlib import Path

from .schema import LABELS


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(data_dir, split, scripts=None):
    """scripts: None/'all' = every script, or a list such as ['en']."""
    rows = read_jsonl(Path(data_dir) / f"d3_{split}.jsonl")
    if scripts and scripts != "all":
        rows = [r for r in rows if r["script"] in scripts]
    return rows


def pred_rows(rows, probs):
    """probs: list of 4-vectors ordered as schema.LABELS -> preds.jsonl records."""
    out = []
    for r, p in zip(rows, probs):
        p = [float(x) for x in p]
        out.append({"id": r["id"], "pred": LABELS[max(range(len(p)), key=p.__getitem__)], "probs": p})
    return out
'''

# ----------------------------------------------------------------------------------------------
FILES["src/swasthya/prompts.py"] = r'''"""Prompt + letter-scoring helpers for the LLM gate (prompted V0 and QLoRA G1).

The model is asked to answer with ONE letter. We never generate: we read the next-token logits of
the four letters at the last position and softmax them -> calibrated-ish probabilities, one forward pass.
The rubric below is deliberately generic (category-level), not copied from any dev/test case.
"""
import random
from collections import defaultdict

from .schema import LABELS

LETTERS = ["A", "B", "C", "D"]          # same order as schema.LABELS

SYSTEM = (
    "You are a triage assistant at the front desk of a public clinic in India. A patient or a relative "
    "sends a short message in English, Hindi (Devanagari) or Hinglish (Hindi in Latin letters). "
    "Classify how urgent it is:\n"
    "A) EMERGENCY - life or limb in immediate danger (for example breathing trouble, collapse or "
    "unresponsiveness, seizure, severe bleeding, poisoning, stroke or heart-attack signs, serious danger "
    "in pregnancy or to a newborn, suicidal intent). Staff must be alerted right now.\n"
    "B) URGENT - needs a doctor the same day but is not immediately life-threatening.\n"
    "C) ROUTINE - can wait for a normal appointment (mild or long-standing problems, check-ups, reports, refills).\n"
    "D) INSUFFICIENT - too vague to judge; more questions are needed.\n"
    "If torn between two levels, choose the more urgent one. Reply with only the letter."
)


def user_prompt(text):
    return f"Patient message:\n{text}\n\nAnswer with a single letter (A, B, C or D)."


def build_messages(text, shots=()):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in shots:
        msgs.append({"role": "user", "content": user_prompt(ex["text"])})
        msgs.append({"role": "assistant", "content": LETTERS[LABELS.index(ex["label"])]})
    msgs.append({"role": "user", "content": user_prompt(text)})
    return msgs


def pick_shots(train_rows, k, seed=13):
    """k//4 examples per label, deterministic, scripts mixed."""
    if k <= 0:
        return []
    by = defaultdict(list)
    for r in sorted(train_rows, key=lambda r: r["id"]):
        by[r["label"]].append(r)
    rng = random.Random(seed)
    shots = []
    for lab in LABELS:
        shots += rng.sample(by[lab], k // len(LABELS))
    rng.shuffle(shots)
    return shots


def render(tok, text, shots=()):
    return tok.apply_chat_template(build_messages(text, shots), tokenize=False, add_generation_prompt=True)


def letter_token_ids(tok):
    ids = []
    for L in LETTERS:
        t = tok.encode(L, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"letter {L!r} is not a single token for this tokenizer: {t}")
        ids.append(t[0])
    return ids


def letter_logits(model, tok, texts, letter_ids, device, shots=(), max_len=768):
    """(B,4) float32 logits for the four letters at the answer position. Needs left padding."""
    import torch
    prompts = [render(tok, t, shots) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
              add_special_tokens=False).to(device)
    use_amp = torch.device(device).type == "cuda"
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=use_amp):
        try:   # only the last position is needed: saves GBs of vocab-sized logits on long few-shot prompts
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            logits = model(**enc).logits[:, -1, :]
    return logits[:, letter_ids].float()
'''

# ----------------------------------------------------------------------------------------------
FILES["kaggle/run_muril.py"] = r'''"""B1 (+ training-script ablation): fine-tune an encoder (MuRIL) for 4-way triage.

Kaggle example:
  python run_muril.py --train_scripts all en deva hinglish --seeds 13 42 2026 --out /kaggle/working/outputs
Quick environment check:  python run_muril.py --smoke
Each run writes <out>/<name>_<train_scripts>_s<seed>/{preds_dev.jsonl,preds_test.jsonl,config.json,log.json}
Hyper-parameters are FIXED (no model selection on dev or test).
"""
import argparse
import glob
import json
import os
import random
import sys
import time

for _p in ["src"] + glob.glob("/kaggle/input/**/src", recursive=True):
    if os.path.isdir(os.path.join(_p, "swasthya")):
        sys.path.insert(0, _p)
        ROOT = os.path.dirname(os.path.abspath(_p))
        break
else:
    raise SystemExit("cannot find src/swasthya (is the dataset attached?)")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from swasthya import kio  # noqa: E402
from swasthya.metrics import emergency_recall  # noqa: E402
from swasthya.schema import LABEL2ID, SCRIPTS  # noqa: E402


def predict(model, tok, rows, dev, max_len, bs=64):
    model.eval()
    out = []
    amp = dev.type == "cuda"
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            enc = tok([r["text"] for r in rows[i:i + bs]], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                logits = model(**enc).logits
            out += torch.softmax(logits.float(), -1).cpu().tolist()
    return out


def run_one(a, ts, seed, dev, data):
    name = f"{a.name}_{ts}_s{seed}"
    run_dir = os.path.join(a.out, name)
    os.makedirs(run_dir, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    scripts = SCRIPTS if ts == "all" else [ts]
    train = [r for r in data["train"] if r["script"] in scripts]
    if a.smoke:
        train = train[:32]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=4).to(dev)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    n_batches = (len(train) + a.bs - 1) // a.bs
    total = a.epochs * n_batches
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    w = torch.tensor([a.emerg_weight, 1.0, 1.0, 1.0], device=dev)
    g = torch.Generator().manual_seed(seed)
    log, t0 = [], time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(train), generator=g).tolist()
        tot = 0.0
        for i in range(0, len(perm), a.bs):
            b = [train[j] for j in perm[i:i + a.bs]]
            enc = tok([r["text"] for r in b], truncation=True, max_length=a.max_len, padding=True,
                      return_tensors="pt").to(dev)
            y = torch.tensor([LABEL2ID[r["label"]] for r in b], device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            loss = F.cross_entropy(logits.float(), y, weight=w)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item() * len(b)
        pd = predict(model, tok, data["dev"], dev, a.max_len)
        pr = [r["pred"] for r in kio.pred_rows(data["dev"], pd)]
        rec = emergency_recall([r["label"] for r in data["dev"]], pr)
        log.append({"epoch": ep + 1, "train_loss": tot / len(train), "dev_emergency_recall": rec})
        print(f"[{name}] epoch {ep + 1}/{a.epochs} loss {tot / len(train):.4f} dev_emRecall {rec:.3f}", flush=True)
    for split in ("dev", "test"):
        probs = predict(model, tok, data[split], dev, a.max_len)
        kio.write_jsonl(os.path.join(run_dir, f"preds_{split}.jsonl"), kio.pred_rows(data[split], probs))
    cfg = {"system": a.name, "model": a.model, "train_scripts": ts, "seed": seed, "epochs": a.epochs,
           "lr": a.lr, "bs": a.bs, "max_len": a.max_len, "emerg_weight": a.emerg_weight,
           "n_train": len(train), "shots": 0, "seconds": round(time.time() - t0, 1)}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    json.dump(log, open(os.path.join(run_dir, "log.json"), "w"), indent=2)
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--name", default="muril")
    ap.add_argument("--train_scripts", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--emerg_weight", type=float, default=1.0)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data/d3_triage/rendered"))
    ap.add_argument("--out", default="/kaggle/working/outputs" if os.path.isdir("/kaggle/working") else "outputs/raw")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {s: kio.load_split(a.data_dir, s) for s in ("train", "dev", "test")}
    if a.smoke:
        a.epochs, a.train_scripts, a.seeds, a.name = 1, a.train_scripts[:1], a.seeds[:1], "smoke_" + a.name
        data["dev"], data["test"] = data["dev"][:16], data["test"][:16]
    print("device", dev, "| train", len(data["train"]), "dev", len(data["dev"]), "test", len(data["test"]))
    for ts in a.train_scripts:
        for seed in a.seeds:
            run_one(a, ts, seed, dev, data)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
'''

# ----------------------------------------------------------------------------------------------
FILES["kaggle/run_llm_prompted.py"] = r'''"""V0: prompted LLM gate (no training). Zero-shot (--shots 0) or few-shot (--shots 8 = 2 per label).

Kaggle example:
  python run_llm_prompted.py --shots 0 --name v0_zeroshot
  python run_llm_prompted.py --shots 8 --name v0_fewshot8
Quick environment check: python run_llm_prompted.py --smoke
"""
import argparse
import glob
import json
import os
import sys
import time

for _p in ["src"] + glob.glob("/kaggle/input/**/src", recursive=True):
    if os.path.isdir(os.path.join(_p, "swasthya")):
        sys.path.insert(0, _p)
        ROOT = os.path.dirname(os.path.abspath(_p))
        break
else:
    raise SystemExit("cannot find src/swasthya (is the dataset attached?)")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from swasthya import kio, prompts  # noqa: E402


def load_llm(model_name, bits):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if bits == 4:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    cuda = torch.cuda.is_available()
    dt = torch.float16 if cuda else torch.float32
    try:  # transformers >= 4.56 / 5.x name it `dtype`; older versions only know `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dt, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dt, **kw)
    if bits != 4:
        model = model.to("cuda" if cuda else "cpu")
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--name", default="v0_zeroshot")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 16])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data/d3_triage/rendered"))
    ap.add_argument("--out", default="/kaggle/working/outputs" if os.path.isdir("/kaggle/working") else "outputs/raw")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = {s: kio.load_split(a.data_dir, s) for s in ("train", "dev", "test")}
    if a.smoke:
        a.name = "smoke_" + a.name
        data["dev"], data["test"] = data["dev"][:16], data["test"][:16]
    shots = prompts.pick_shots(data["train"], a.shots, seed=13)
    tok, model = load_llm(a.model, a.bits)
    model.eval()
    letter_ids = prompts.letter_token_ids(tok)
    run_dir = os.path.join(a.out, a.name)
    t0 = time.time()
    for split in ("dev", "test"):
        rows, probs = data[split], []
        with torch.no_grad():
            for i in range(0, len(rows), a.bs):
                lg = prompts.letter_logits(model, tok, [r["text"] for r in rows[i:i + a.bs]], letter_ids,
                                           dev, shots=shots, max_len=a.max_len)
                probs += torch.softmax(lg, -1).cpu().tolist()
        kio.write_jsonl(os.path.join(run_dir, f"preds_{split}.jsonl"), kio.pred_rows(rows, probs))
        print(f"[{a.name}] {split}: {len(rows)} rows done", flush=True)
    cfg = {"system": a.name, "model": a.model, "train_scripts": "-", "seed": "-", "shots": a.shots,
           "shot_ids": [s["id"] for s in shots], "bits": a.bits, "seconds": round(time.time() - t0, 1)}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    print("done ->", run_dir)


if __name__ == "__main__":
    main()
'''

# ----------------------------------------------------------------------------------------------
FILES["kaggle/run_qlora_gate.py"] = r'''"""G1: QLoRA-fine-tuned safety gate (Qwen2.5-3B-Instruct, 4-bit) trained on D3 train.

The LM head is used as a 4-way classifier: logits of the letters A-D at the answer position,
cross-entropy over those 4 (same read-out as the prompted V0, so V0 vs G1 is a clean comparison).

Kaggle example:
  python run_qlora_gate.py --train_scripts all --seed 13 --name qlora
Ablation (cross-script transfer):  python run_qlora_gate.py --train_scripts en --seed 13 --name qlora
Quick environment check: python run_qlora_gate.py --smoke
"""
import argparse
import glob
import json
import os
import random
import sys
import time

for _p in ["src"] + glob.glob("/kaggle/input/**/src", recursive=True):
    if os.path.isdir(os.path.join(_p, "swasthya")):
        sys.path.insert(0, _p)
        ROOT = os.path.dirname(os.path.abspath(_p))
        break
else:
    raise SystemExit("cannot find src/swasthya (is the dataset attached?)")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402

from swasthya import kio, prompts  # noqa: E402
from swasthya.metrics import emergency_recall  # noqa: E402
from swasthya.schema import LABEL2ID, SCRIPTS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_llm_prompted import load_llm  # noqa: E402


def predict(model, tok, rows, letter_ids, dev, max_len, bs=8):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            lg = prompts.letter_logits(model, tok, [r["text"] for r in rows[i:i + bs]], letter_ids, dev,
                                       max_len=max_len)
            out += torch.softmax(lg, -1).cpu().tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--name", default="qlora")
    ap.add_argument("--train_scripts", default="all")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 16])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--emerg_weight", type=float, default=1.0)
    ap.add_argument("--save_adapter", action="store_true")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data/d3_triage/rendered"))
    ap.add_argument("--out", default="/kaggle/working/outputs" if os.path.isdir("/kaggle/working") else "outputs/raw")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    scripts = SCRIPTS if a.train_scripts == "all" else [a.train_scripts]
    data = {s: kio.load_split(a.data_dir, s) for s in ("train", "dev", "test")}
    train = [r for r in data["train"] if r["script"] in scripts]
    if a.smoke:
        a.name, a.epochs = "smoke_" + a.name, 1
        train = train[:16]
        data["dev"], data["test"] = data["dev"][:16], data["test"][:16]
    name = f"{a.name}_{a.train_scripts}_s{a.seed}"
    run_dir = os.path.join(a.out, name)
    print("device", dev, "| train", len(train), "dev", len(data["dev"]), "test", len(data["test"]), flush=True)

    tok, model = load_llm(a.model, a.bits)
    model.config.use_cache = False
    if a.bits == 4:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    letter_ids = prompts.letter_token_ids(tok)

    params = [p for p in model.parameters() if p.requires_grad]
    micro = (len(train) + a.bs - 1) // a.bs
    total_steps = a.epochs * ((micro + a.accum - 1) // a.accum)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, max(1, int(0.05 * total_steps)), total_steps)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    w = torch.tensor([a.emerg_weight, 1.0, 1.0, 1.0], device=dev)
    g = torch.Generator().manual_seed(a.seed)
    log, t0 = [], time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(train), generator=g).tolist()
        tot = 0.0
        for step, i in enumerate(range(0, len(perm), a.bs)):
            b = [train[j] for j in perm[i:i + a.bs]]
            y = torch.tensor([LABEL2ID[r["label"]] for r in b], device=dev)
            lg = prompts.letter_logits(model, tok, [r["text"] for r in b], letter_ids, dev, max_len=a.max_len)
            loss = F.cross_entropy(lg, y, weight=w)
            scaler.scale(loss / a.accum).backward()
            tot += loss.item() * len(b)
            if (step + 1) % a.accum == 0 or step + 1 == micro:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        pd = predict(model, tok, data["dev"], letter_ids, dev, a.max_len)
        pr = [r["pred"] for r in kio.pred_rows(data["dev"], pd)]
        rec = emergency_recall([r["label"] for r in data["dev"]], pr)
        log.append({"epoch": ep + 1, "train_loss": tot / len(train), "dev_emergency_recall": rec,
                    "elapsed_s": round(time.time() - t0)})
        print(f"[{name}] epoch {ep + 1}/{a.epochs} loss {tot / len(train):.4f} dev_emRecall {rec:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    for split in ("dev", "test"):
        probs = predict(model, tok, data[split], letter_ids, dev, a.max_len)
        kio.write_jsonl(os.path.join(run_dir, f"preds_{split}.jsonl"), kio.pred_rows(data[split], probs))
    cfg = {"system": a.name, "model": a.model, "train_scripts": a.train_scripts, "seed": a.seed, "epochs": a.epochs,
           "lr": a.lr, "bs": a.bs, "accum": a.accum, "lora_r": a.r, "emerg_weight": a.emerg_weight,
           "n_train": len(train), "shots": 0, "bits": a.bits, "seconds": round(time.time() - t0, 1)}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    json.dump(log, open(os.path.join(run_dir, "log.json"), "w"), indent=2)
    if a.save_adapter:
        model.save_pretrained(os.path.join(run_dir, "adapter"))
    print("done ->", run_dir)


if __name__ == "__main__":
    main()
'''

# ----------------------------------------------------------------------------------------------
FILES["scripts/prepare_train.py"] = r'''"""Validate d3_train.csv, check overlap with dev/test, export data/d3_triage/rendered/d3_train.jsonl.

Run from repo root (after validate_d3.py --export):  python scripts/prepare_train.py [--drop-sim 0.8]
Train cases whose text is a near-copy (char 5-gram Jaccard >= drop-sim, in ANY script) of a dev/test
rendering are dropped from the export (whole case, so scripts stay parallel).
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import d3, kio  # noqa: E402
from swasthya.keyword_baseline import norm  # noqa: E402
from swasthya.schema import SCRIPTS  # noqa: E402

REND = ROOT / "data/d3_triage/rendered"


def grams(t, n=5):
    t = f" {norm(t)} "
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/d3_triage/human_cases/d3_train.csv"))
    ap.add_argument("--drop-sim", type=float, default=0.8)
    a = ap.parse_args()

    rows = d3.load_csv(a.csv)
    print(f"{Path(a.csv).name}: {len(rows)} cases")
    errors, warnings = d3.validate_rows(rows)
    for r in rows:
        if not r["case_id"].upper().startswith("T"):
            warnings.append(f"{r['case_id']}: id does not start with T")
    for w in warnings[:20]:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    if errors:
        print(f"\n{len(errors)} error(s). Fix d3_train.csv (or ask the writer) and re-run.")
        return 1
    print("labels", dict(Counter(r["label"] for r in rows)))
    print("categories", dict(Counter(r["category"] for r in rows)))

    gold = []
    for sp in ("dev", "test"):
        p = REND / f"d3_{sp}.jsonl"
        if not p.exists():
            print("Missing", p, "- run: python scripts/validate_d3.py --export")
            return 1
        gold += kio.read_jsonl(p)
    gold_grams = {s: [(g["id"], grams(g["text"])) for g in gold if g["script"] == s] for s in SCRIPTS}

    case_max, matches = {}, []
    for r in rows:
        for s in SCRIPTS:
            tg = grams(r[s])
            best_id, best = max(((gid, jac(tg, gg)) for gid, gg in gold_grams[s]), key=lambda x: x[1])
            case_max[r["case_id"]] = max(case_max.get(r["case_id"], 0.0), best)
            if best >= 0.4:
                matches.append((r["case_id"], s, round(best, 3), best_id))
    sims = sorted(case_max.values())
    print(f"\nTrain-vs-eval similarity (max over scripts): mean {sum(sims) / len(sims):.3f} "
          f"max {sims[-1]:.3f} | >=0.5: {sum(x >= 0.5 for x in sims)} | >={a.drop_sim}: {sum(x >= a.drop_sim for x in sims)}")
    gold_cues = {tuple(sorted(set(g["cues"]))) for g in gold}
    same_cues = sum(tuple(d3.parse_cues(r["cues"])) in gold_cues for r in rows)
    print(f"train cases whose exact cue-set also occurs in dev/test: {same_cues}/{len(rows)}")
    out_csv = ROOT / "outputs/results/train_eval_similarity.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train_case", "script", "jaccard5", "closest_eval_id"])
        w.writerows(sorted(matches, key=lambda m: -m[2]))
    print("saved", out_csv.relative_to(ROOT), f"({len(matches)} pairs with sim >= 0.4)")

    drop = {c for c, m in case_max.items() if m >= a.drop_sim}
    if drop:
        print("DROPPED (near-copies of dev/test):", sorted(drop))
    out = []
    for r in rows:
        if r["case_id"] in drop:
            continue
        for s in SCRIPTS:
            out.append({"id": f"{r['case_id']}|{s}", "case_id": r["case_id"], "script": s, "text": r[s],
                        "label": r["label"], "category": r["category"], "pair_id": None, "split": "train",
                        "cues": d3.parse_cues(r["cues"])})
    kept = len(rows) - len(drop)
    if kept < 0.8 * len(rows):
        print(f"\nERROR: {len(drop)} of {len(rows)} train cases would be dropped as near-copies of dev/test.")
        print("That is too many: look at outputs/results/train_eval_similarity.csv, talk to the writer, or raise --drop-sim.")
        return 1
    kio.write_jsonl(REND / "d3_train.jsonl", out)
    print(f"\nexported {len(out)} renderings ({kept} cases) -> data/d3_triage/rendered/d3_train.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# ----------------------------------------------------------------------------------------------
FILES["scripts/make_kaggle_bundle.py"] = r'''"""Bundle code + data for upload to Kaggle as ONE dataset.

Run from repo root:  python scripts/make_kaggle_bundle.py [--kaggle-user YOURNAME]
Creates kaggle_bundle/swasthya-data/ (folder) and kaggle_bundle/swasthya-data.zip (same content at zip top level).
Re-run + upload as a NEW VERSION whenever src/, kaggle/ or the data changes.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-user", default=None, help="writes dataset-metadata.json for the kaggle CLI")
    a = ap.parse_args()
    need = [ROOT / f"data/d3_triage/rendered/d3_{s}.jsonl" for s in ("train", "dev", "test")]
    miss = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if miss:
        print("Missing:", miss, "\n-> run validate_d3.py --export and prepare_train.py first")
        return 1
    out = ROOT / "kaggle_bundle" / "swasthya-data"
    if out.exists():
        shutil.rmtree(out)
    (out / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src/swasthya", out / "src/swasthya", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "kaggle", out / "kaggle", ignore=shutil.ignore_patterns("__pycache__", ".gitkeep"))
    (out / "data/d3_triage/rendered").mkdir(parents=True)
    for p in need:
        shutil.copy(p, out / "data/d3_triage/rendered" / p.name)
    if a.kaggle_user:
        (out / "dataset-metadata.json").write_text(json.dumps(
            {"title": "swasthya-data", "id": f"{a.kaggle_user}/swasthya-data",
             "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    zp = ROOT / "kaggle_bundle" / "swasthya-data.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out))
    print("bundle folder:", out.relative_to(ROOT))
    print("bundle zip   :", zp.relative_to(ROOT), f"({zp.stat().st_size / 1e3:.0f} KB)")
    print("Upload either one as a Kaggle Dataset named swasthya-data (see docs/KAGGLE_RUNBOOK.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# ----------------------------------------------------------------------------------------------
FILES["scripts/score_all.py"] = r'''"""Score every preds file under outputs/raw/ (+ the keyword baseline) and build one summary table.

Run from repo root after copying Kaggle outputs into outputs/raw/:  python scripts/score_all.py [--boot 1000]
Writes outputs/results/<run>_<split>.json for every run and outputs/results/summary.csv
Runs whose preds do not cover every gold id (e.g. smoke tests) are skipped.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import evaluate as ev  # noqa: E402
from swasthya.keyword_baseline import predict  # noqa: E402


def f3(x):
    return "" if x is None or x != x else round(float(x), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--raw", default="outputs/raw")
    a = ap.parse_args()
    gold = {s: ev.load_jsonl(ROOT / f"data/d3_triage/rendered/d3_{s}.jsonl") for s in ("dev", "test")}
    jobs = []
    for sp, g in gold.items():
        jobs.append(("b0_keyword", {"system": "keyword", "train_scripts": "-", "seed": "-", "shots": "-"}, sp,
                     {x["id"]: {"pred": predict(x["text"])} for x in g}))
    for pf in sorted((ROOT / a.raw).rglob("preds_*.jsonl")):
        sp = pf.stem.split("_", 1)[1]
        if sp not in gold or pf.parent.name.startswith("smoke"):
            continue
        cfgp = pf.parent / "config.json"
        cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {"system": pf.parent.name}
        jobs.append((pf.parent.name, cfg, sp, {r["id"]: r for r in ev.load_jsonl(pf)}))
    rows = []
    for run, cfg, sp, preds in jobs:
        try:
            res = ev.evaluate(gold[sp], preds, n_boot=a.boot)
        except ValueError as e:
            print("SKIP", run, sp, "-", str(e)[:90])
            continue
        (ROOT / "outputs/results").mkdir(parents=True, exist_ok=True)
        (ROOT / f"outputs/results/{run}_{sp}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        o, lo_hi = res["overall"], res["overall"]["emergency_recall_ci"]
        rec, mp = res["emergency_recall_by_script"], res["minimal_pairs"]
        rows.append({
            "run": run, "system": cfg.get("system"), "train_scripts": cfg.get("train_scripts"), "seed": cfg.get("seed"),
            "shots": cfg.get("shots"), "split": sp, "n": o["n"], "em_recall": f3(o["emergency_recall"]),
            "em_ci_lo": f3(lo_hi[0]), "em_ci_hi": f3(lo_hi[1]), "over_triage": f3(o["over_triage"]),
            "under_triage": f3(o["under_triage"]), "macro_f1": f3(o["macro_f1"]), "ece": f3(o.get("ece")),
            "script_gap": f3(res["cross_script_recall_gap"]),
            **{f"recall_{s}": f3(rec.get(s)) for s in ("en", "deva", "hinglish")},
            **{f"flip_{s}": f3(mp[s]["flip_acc"]) for s in ("en", "deva", "hinglish")}})
    out = ROOT / "outputs/results/summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'run':<26}{'split':<6}{'emRec':>7}{'[CI]':>16}{'over':>7}{'under':>7}{'mF1':>7}{'gap':>7}")
    for r in sorted(rows, key=lambda r: (r["split"], r["run"])):
        ci = f"[{r['em_ci_lo']},{r['em_ci_hi']}]"
        print(f"{r['run']:<26}{r['split']:<6}{r['em_recall']:>7}{ci:>16}{r['over_triage']:>7}{r['under_triage']:>7}"
              f"{r['macro_f1']:>7}{r['script_gap']:>7}")
    print("\nsaved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
'''

# ----------------------------------------------------------------------------------------------
FILES["tests/test_prompts.py"] = r'''from swasthya import kio, prompts
from swasthya.schema import LABELS


def test_letters_match_labels():
    assert len(prompts.LETTERS) == len(LABELS) == 4


def test_messages_zero_and_few_shot():
    m = prompts.build_messages("hello")
    assert [x["role"] for x in m] == ["system", "user"] and "hello" in m[-1]["content"]
    m = prompts.build_messages("q", shots=[{"text": "x", "label": "ROUTINE"}])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]
    assert m[2]["content"] == "C"


def test_pick_shots_balanced_and_deterministic():
    rows = [{"id": f"{l}{i}", "text": "t", "label": l} for l in LABELS for i in range(10)]
    a, b = prompts.pick_shots(rows, 8, 13), prompts.pick_shots(rows, 8, 13)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert sorted(r["label"] for r in a) == sorted(LABELS * 2)
    assert prompts.pick_shots(rows, 0) == []


def test_pred_rows_argmax():
    rows = [{"id": "a"}, {"id": "b"}]
    out = kio.pred_rows(rows, [[0.1, 0.7, 0.1, 0.1], [0.9, 0.05, 0.03, 0.02]])
    assert [o["pred"] for o in out] == ["URGENT", "EMERGENCY"]
'''

# ----------------------------------------------------------------------------------------------
FILES["kaggle/.gitkeep"] = ""

FILES["docs/KAGGLE_RUNBOOK.md"] = r'''# Kaggle runbook (all GPU work)

## 0. Local prep (once per data/code change)
```bash
python scripts/validate_d3.py --export        # eval splits
python scripts/prepare_train.py               # train split + overlap check
python scripts/make_kaggle_bundle.py          # -> kaggle_bundle/swasthya-data.zip
```

## 1. Kaggle one-time setup
1. kaggle.com -> Datasets -> New Dataset -> upload `kaggle_bundle/swasthya-data.zip` -> title **swasthya-data** -> Create (private).
   Code/data changed later? Dataset page -> **New Version** -> upload the new zip.
2. New Notebook -> **Add Input** -> your swasthya-data dataset.
3. Right panel: **Accelerator = GPU T4 x2** (use one GPU; avoid P100 for 4-bit), **Internet = On** (needs phone verification), Persistence: none.

## 2. Notebook cells (run in order; do the --smoke line of each script FIRST)
```
# cell 1: install
!pip install -q -U bitsandbytes peft accelerate

# cell 2: locate scripts
KD = !find /kaggle/input -type d -name kaggle | head -1
KD = KD[0]; print(KD)

# cell 3: MuRIL baseline + train-script ablation: 4 train settings x 3 seeds (~15-25 min total)
!python $KD/run_muril.py --smoke
!python $KD/run_muril.py --train_scripts all en deva hinglish --seeds 13 42 2026

# cell 4: prompted LLM (V0), zero-shot and 8-shot (~10-15 min each)
!python $KD/run_llm_prompted.py --smoke
!python $KD/run_llm_prompted.py --shots 0 --name v0_zeroshot
!python $KD/run_llm_prompted.py --shots 8 --name v0_fewshot8

# cell 5: QLoRA gate (~30-60 min per run)
!python $KD/run_qlora_gate.py --smoke
!python $KD/run_qlora_gate.py --train_scripts all --seed 13 --name qlora
# ablation, only if time/quota allows:
!python $KD/run_qlora_gate.py --train_scripts en --seed 13 --name qlora

# cell 6: package outputs
!cd /kaggle/working && zip -rq outputs.zip outputs && ls -la outputs.zip
```
Download `outputs.zip` from the notebook's **Output** tab (after "Save Version" -> "Save & Run All", or from the file browser on the right).
Kaggle sessions die after ~9-12 h and you get ~30 GPU-h/week: run in the order above and zip after every cell group.

## 3. Back on your PC
```bash
unzip -q ~/Downloads/outputs.zip -d /tmp/k && cp -r /tmp/k/outputs/* outputs/raw/
python scripts/score_all.py                  # -> outputs/results/summary.csv (+ one JSON per run)
```
Then add rows to `docs/EXPERIMENT_LOG.md` and commit `outputs/results/` (raw preds stay out of git).

## 4. Experiment IDs (use in the report)
| ID | System | What varies |
|----|--------|-------------|
| B0 | keyword lexicon | - |
| B1 | MuRIL, trained on all scripts, 3 seeds | baseline |
| A1 | MuRIL trained on en / deva / hinglish only | cross-script transfer (ablation) |
| V0 / V0-fs | Qwen2.5-3B prompted, 0 / 8 shots | no training |
| G1 | Qwen2.5-3B + QLoRA, all scripts | our gate |
| A2 | G1 trained on en only | cross-script transfer (ablation) |
'''


def main():
    root = Path(".")
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
