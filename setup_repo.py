"""Creates the Swasthya Saathi repo skeleton.

Usage:  python setup_repo.py            (run inside your empty repo folder)
Safe to re-run: existing files are never overwritten.
"""
from pathlib import Path

DIRS = [
    "configs",
    "data/d3_triage/specs",      # structured case specs (source of truth)
    "data/d3_triage/rendered",   # en / devanagari / hinglish renderings
    "data/d4_clinic",            # clinic simulator scenarios + synthetic DB
    "docs",
    "kaggle",                    # self-contained scripts/notebooks to upload
    "outputs/raw",               # files downloaded from Kaggle (gitignored if big)
    "outputs/results",           # small result JSON/CSV you DO commit
    "outputs/figures",
    "report",                    # mid-submission + final report
    "scripts",
    "src/swasthya",
    "tests",
]

FILES = {}

FILES[".gitignore"] = """__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
outputs/raw/
*.safetensors
*.bin
*.pt
.env
"""

FILES["requirements.txt"] = """numpy
pandas
scikit-learn
scipy
statsmodels
matplotlib
pyyaml
tqdm
jsonschema
# GPU-side (installed on Kaggle, optional locally):
# torch, transformers, peft, bitsandbytes, accelerate, datasets
"""

FILES["README.md"] = """# Swasthya Saathi
Safety-gated, Hindi-first clinic front-desk agent (LMA Group Project 2026).

Core idea: a fine-tuned **safety gate** decides urgency, a deterministic **action guard**
approves state-changing tool calls; the LLM only coordinates dialogue.

## Layout
- `src/swasthya/`  shared library (labels, schemas, metrics, clinic sim, guard)
- `data/d3_triage/` script-parallel triage set (specs -> en/deva/hinglish)
- `data/d4_clinic/` clinic simulator scenarios
- `kaggle/`        standalone scripts you upload to Kaggle (GPU work)
- `outputs/`       results downloaded from Kaggle
- `docs/`          workflow, decisions, progress log
- `report/`        mid-submission / final report

## Workflow
See `docs/WORKFLOW.md` (local = write/CPU/analysis, Kaggle = GPU training/inference).
"""

FILES["docs/WORKFLOW.md"] = """# Local <-> Kaggle workflow
1. Write + test code locally (CPU, tiny samples).
2. Zip `src/` + needed `data/` files -> upload as a Kaggle Dataset (version it: v1, v2...).
3. Kaggle notebook: attach the dataset, run a script from `kaggle/`, write outputs to `/kaggle/working/`.
4. Download outputs -> put in `outputs/raw/<experiment_name>/`.
5. Small metric files -> copy to `outputs/results/` and commit.
6. Every experiment gets a row in `docs/EXPERIMENT_LOG.md`.

Rules: fixed seeds (13, 42, 2026); every Kaggle script reads a config in `configs/`
and saves `config + git-commit/dataset-version + metrics.json` next to its outputs.
"""

FILES["docs/EXPERIMENT_LOG.md"] = """# Experiment log
| ID | Date | What | Config | Dataset ver. | Result file | Notes |
|----|------|------|--------|--------------|-------------|-------|
"""

FILES["docs/PROGRESS.md"] = """# Progress log (feeds the mid-submission)
## Done
- [ ] Repo scaffold
## In progress
## Challenges + mitigations (fill as they happen)
| Challenge | Impact | Mitigation |
|-----------|--------|-----------|
"""

FILES["configs/default.yaml"] = """seeds: [13, 42, 2026]
labels: [EMERGENCY, URGENT, ROUTINE, INSUFFICIENT]
scripts: [en, deva, hinglish]
splits: {train: 0.6, dev: 0.15, test: 0.25}
"""

FILES["src/swasthya/__init__.py"] = ""

FILES["src/swasthya/schema.py"] = '''"""Shared labels and the structured case specification (D3)."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional

LABELS = ["EMERGENCY", "URGENT", "ROUTINE", "INSUFFICIENT"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
SCRIPTS = ["en", "deva", "hinglish"]
CATEGORIES = [
    "airway_breathing_consciousness", "heart_brain", "bleeding_injury_poisoning",
    "maternal", "newborn_child", "general_medical", "mental_health_crisis",
]


@dataclass
class CaseSpec:
    case_id: str
    category: str
    label: str                       # gold urgency label
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: List[str] = field(default_factory=list)
    duration: Optional[str] = None
    severity: Optional[str] = None
    red_flag_cues: List[str] = field(default_factory=list)  # cue checklist
    pair_id: Optional[str] = None    # minimal-pair group
    split: str = "train"

    def to_dict(self):
        return asdict(self)
'''

FILES["src/swasthya/metrics.py"] = '''"""Triage metrics. Primary: emergency recall; also under/over-triage, macro-F1, ECE."""
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
'''

FILES["tests/test_metrics.py"] = '''from swasthya.metrics import emergency_recall, over_triage_rate, macro_f1


def test_emergency_recall():
    y = ["EMERGENCY", "EMERGENCY", "ROUTINE"]
    p = ["EMERGENCY", "ROUTINE", "ROUTINE"]
    assert emergency_recall(y, p) == 0.5
    assert over_triage_rate(y, p) == 0.0


def test_macro_f1_perfect():
    y = ["EMERGENCY", "URGENT", "ROUTINE", "INSUFFICIENT"]
    assert macro_f1(y, y) == 1.0
'''


def main():
    root = Path(".")
    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
        keep = root / d / ".gitkeep"
        if not any((root / d).iterdir()):
            keep.touch()
    for path, content in FILES.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            print("skip   ", path)
        else:
            p.write_text(content, encoding="utf-8")
            print("created", path)
    print("\nDone. Next: python -m venv .venv && pip install -r requirements.txt pytest")
    print("Then:   set PYTHONPATH=src  (Windows)  /  export PYTHONPATH=src  (Linux/Mac)  &&  pytest -q")


if __name__ == "__main__":
    main()
