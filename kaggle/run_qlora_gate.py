"""G1: QLoRA-fine-tuned safety gate (Qwen2.5-3B-Instruct, 4-bit) trained on D3 train.

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
