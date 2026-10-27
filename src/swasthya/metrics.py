"""Triage metrics. Primary: emergency recall; also under/over-triage, macro-F1, ECE."""
import numpy as np
from .schema import LABELS, LABEL2ID

SEVERITY = {"EMERGENCY": 3, "URGENT": 2, "ROUTINE": 1, "INSUFFICIENT": 0}


def emergency_recall(y_true, y_pred):
    idx = [i for i, y in enumerate(y_true) if y == "EMERGENCY"]
    if not idx:
        return float("nan")
    return float(np.mean([y_pred[i] == "EMERGENCY" for i in idx]))


def over_triage_rate(y_true, y_pred):
    """Share of non-emergency cases predicted EMERGENCY."""
    idx = [i for i, y in enumerate(y_true) if y != "EMERGENCY"]
    if not idx:
        return float("nan")
    return float(np.mean([y_pred[i] == "EMERGENCY" for i in idx]))


def under_triage_rate(y_true, y_pred):
    """Share of cases (gold URGENT/EMERGENCY) predicted less severe than gold."""
    idx = [i for i, y in enumerate(y_true) if y in ("EMERGENCY", "URGENT")]
    if not idx:
        return float("nan")
    return float(np.mean([SEVERITY[y_pred[i]] < SEVERITY[y_true[i]] for i in idx]))


def macro_f1(y_true, y_pred):
    f1s = []
    for l in LABELS:
        tp = sum(t == l and p == l for t, p in zip(y_true, y_pred))
        fp = sum(t != l and p == l for t, p in zip(y_true, y_pred))
        fn = sum(t == l and p != l for t, p in zip(y_true, y_pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def ece(probs, y_true, n_bins=10):
    """probs: (N, 4) array ordered as LABELS."""
    probs = np.asarray(probs)
    conf = probs.max(1)
    pred = probs.argmax(1)
    gold = np.array([LABEL2ID[y] for y in y_true])
    acc = (pred == gold).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def bootstrap_ci(metric_fn, y_true, y_pred, groups=None, n=1000, seed=0):
    """Bootstrap CI. Pass groups=case_ids so script renderings of one case resample together."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.array(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    stats = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        stats.append(metric_fn(list(y_true[idx]), list(y_pred[idx])))
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def cross_script_recall_gap(y_true, y_pred, scripts):
    """Max - min emergency recall across scripts."""
    recs = {}
    for s in set(scripts):
        idx = [i for i, x in enumerate(scripts) if x == s]
        recs[s] = emergency_recall([y_true[i] for i in idx], [y_pred[i] for i in idx])
    vals = [v for v in recs.values() if v == v]
    return (max(vals) - min(vals)) if vals else float("nan"), recs
