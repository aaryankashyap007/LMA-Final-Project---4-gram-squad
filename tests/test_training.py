from collections import Counter

import numpy as np
import pytest

from swasthya.metrics import recall_at_overtriage
from swasthya.schema import LABELS, SCRIPTS
from swasthya.training import BestTracker, select_train_rows


def train_rows(n_per_label=10):
    rows = []
    for lab in LABELS:
        for i in range(n_per_label):
            for s in SCRIPTS:
                rows.append({"id": f"{lab[:1]}{i:02d}|{s}", "case_id": f"{lab[:1]}{i:02d}", "script": s,
                             "label": lab, "text": f"{lab} {i} {s}"})
    return rows


def test_select_all_and_single_script():
    rows = train_rows()
    assert len(select_train_rows(rows, "all")) == 120
    en = select_train_rows(rows, "en")
    assert len(en) == 40 and {r["script"] for r in en} == {"en"}
    with pytest.raises(ValueError):
        select_train_rows(rows, "klingon")


def test_mixed1_is_matched_size_with_every_case_once_and_all_scripts_present():
    rows = train_rows()
    m = select_train_rows(rows, "mixed1", seed=13)
    assert len(m) == len(select_train_rows(rows, "en")) == 40             # same size as a single-script run
    assert len({r["case_id"] for r in m}) == 40                           # every case exactly once
    counts = Counter(r["script"] for r in m)
    assert set(counts) == set(SCRIPTS) and max(counts.values()) - min(counts.values()) <= 1
    for lab in LABELS:                                                    # balanced inside each label too
        c = Counter(r["script"] for r in m if r["label"] == lab)
        assert max(c.values()) - min(c.values()) <= 1


def test_mixed1_is_deterministic_and_seed_dependent():
    rows = train_rows()
    a = [r["id"] for r in select_train_rows(rows, "mixed1", seed=13)]
    assert a == [r["id"] for r in select_train_rows(rows, "mixed1", seed=13)]
    assert a != [r["id"] for r in select_train_rows(rows, "mixed1", seed=42)]


def test_best_tracker_ties_keep_the_earlier_epoch_and_patience_stops():
    t = BestTracker(patience=2)
    assert t.update(1, 0.5, 0.6) and t.update(2, 0.7, 0.5)
    assert not t.update(3, 0.7, 0.5) and t.best_epoch == 2                 # exact tie: keep epoch 2
    assert not t.should_stop
    assert not t.update(4, 0.6, 0.9) and t.should_stop                     # two epochs without gain
    t2 = BestTracker(patience=1)
    assert t2.update(1, 0.7, 0.5) and t2.update(2, 0.7, 0.6) and t2.best_epoch == 2   # secondary breaks the tie
    assert not BestTracker(patience=0).should_stop


def test_best_tracker_treats_nan_as_worst():
    t = BestTracker()
    t.update(1, float("nan"))
    assert t.update(2, 0.1) and t.best_epoch == 2


def test_recall_at_overtriage_uses_the_smallest_threshold_within_budget():
    y = ["EMERGENCY"] * 4 + ["ROUTINE"] * 6
    p = [0.9, 0.8, 0.4, 0.2, 0.7, 0.3, 0.1, 0.1, 0.05, 0.05]
    rec, t, over = recall_at_overtriage(y, p, 0.0)
    assert (rec, over) == (0.5, 0.0) and t == pytest.approx(0.8)           # must sit above the 0.7 false alarm
    rec, t, over = recall_at_overtriage(y, p, 1 / 6)
    assert (rec, over) == (0.75, pytest.approx(1 / 6)) and t == pytest.approx(0.4)
    assert recall_at_overtriage(y, p, 1.0)[0] == 1.0
    assert np.isnan(recall_at_overtriage(["ROUTINE"] * 3, [0.1, 0.2, 0.3], 0.1)[0])
