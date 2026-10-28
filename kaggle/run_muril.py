"""B1 (+ training-script ablation): fine-tune an encoder (MuRIL) for 4-way triage.

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
