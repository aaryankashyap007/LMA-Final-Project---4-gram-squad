"""Torch-free helpers for QLoRA gate training: which rows to train on, and best-epoch tracking."""
import random

from .schema import SCRIPTS


def select_train_rows(rows, train_scripts, seed=13):
    """Rows to train on.
      "all"            every rendering (3 per case)
      "en"/"deva"/...  one script only (1 per case)
      "mixed1"         matched-size control: every case appears in exactly ONE script, assigned round-robin
                       (balanced overall and within each label), so the row count equals a single-script run while
                       all scripts are present. Comparing it with a single-script run separates script coverage
                       from data quantity.
    """
    if train_scripts == "all":
        return list(rows)
    if train_scripts in SCRIPTS:
        return [r for r in rows if r["script"] == train_scripts]
    if train_scripts != "mixed1":
        raise ValueError(f"unknown train_scripts {train_scripts!r}")
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["script"]] = r
    label = {cid: next(iter(d.values()))["label"] for cid, d in by_case.items()}
    rng, out, k = random.Random(seed), [], 0
    for lab in sorted(set(label.values())):
        cids = sorted(c for c in by_case if label[c] == lab)
        rng.shuffle(cids)
        for cid in cids:
            s = SCRIPTS[k % len(SCRIPTS)]
            k += 1
            if s in by_case[cid]:
                out.append(by_case[cid][s])
    return sorted(out, key=lambda r: r["id"])


class BestTracker:
    """Best epoch by (primary, secondary). Only a strictly better epoch replaces the best, so ties keep the earlier
    epoch. patience > 0 asks to stop after that many epochs without improvement."""

    def __init__(self, patience=0):
        self.patience, self.best, self.best_epoch, self.since = patience, None, None, 0

    def update(self, epoch, primary, secondary=0.0):
        key = (-1.0 if primary != primary else primary, -1.0 if secondary != secondary else secondary)
        if self.best is None or key > self.best:
            self.best, self.best_epoch, self.since = key, epoch, 0
            return True
        self.since += 1
        return False

    @property
    def should_stop(self):
        return self.patience > 0 and self.since >= self.patience
