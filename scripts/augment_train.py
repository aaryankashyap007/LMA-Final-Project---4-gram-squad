"""Make a noisy-romanisation copy of the training set (train data only; dev/test are never touched).

Run from repo root:  python scripts/augment_train.py [--frac 0.5] [--seed 13]
Reads data/d3_triage/rendered/d3_train.jsonl and writes d3_train_noisy.jsonl next to it: the SAME rows and ids, but a
seeded fraction of the Hinglish renderings get spelling noise (swasthya.noise). Row count and labels are unchanged,
so training on it versus the clean file isolates the effect of the noise from the amount of data.
Then rebuild the Kaggle bundle (it picks up d3_train_*.jsonl) and train with:
  python run_qlora_gate.py --train_scripts all --seed 13 --train_file d3_train_noisy.jsonl --name qlora_noisy
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swasthya import kio  # noqa: E402
from swasthya.noise import noisify  # noqa: E402

REND = ROOT / "data/d3_triage/rendered"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REND / "d3_train.jsonl"))
    ap.add_argument("--out", default=str(REND / "d3_train_noisy.jsonl"))
    ap.add_argument("--frac", type=float, default=0.5, help="share of Hinglish rows that get noise")
    ap.add_argument("--p_word", type=float, default=0.35)
    ap.add_argument("--p_join", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rows = kio.read_jsonl(a.src)
    hing = sorted(r["id"] for r in rows if r["script"] == "hinglish")
    pick = set(random.Random(a.seed).sample(hing, round(a.frac * len(hing))))
    out, changed, shown = [], 0, 0
    for r in rows:
        r = dict(r)
        if r["id"] in pick:
            new = noisify(r["text"], random.Random(f"{a.seed}|{r['id']}"), a.p_word, a.p_join)
            if new != r["text"]:
                if shown < 6:
                    print(f"  {r['text']}\n  -> {new}\n")
                    shown += 1
                r["text"], r["noised"] = new, True
                changed += 1
        out.append(r)
    kio.write_jsonl(a.out, out)
    print(f"{len(rows)} rows ({len(hing)} Hinglish), {len(pick)} selected, {changed} actually changed -> {a.out}")


if __name__ == "__main__":
    main()
