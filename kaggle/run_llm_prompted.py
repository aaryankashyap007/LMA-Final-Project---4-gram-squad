"""V0: prompted LLM gate (no training). Zero-shot (--shots 0) or few-shot (--shots 8 = 2 per label).

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
