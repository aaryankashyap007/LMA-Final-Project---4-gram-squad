"""Small IO helpers shared by Kaggle scripts and local scoring (no torch dependency)."""
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
