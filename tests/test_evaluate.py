import pytest
from swasthya.keyword_baseline import predict
from swasthya import evaluate as ev


def test_keyword_basic():
    assert predict("I cannot breathe") == "EMERGENCY"
    assert predict("मेरे पिताजी को सीने में दर्द है") == "EMERGENCY"
    assert predict("mere pitaji ko seene mein dard hai") == "EMERGENCY"
    assert predict("my son has fever for 3 days") == "URGENT"
    assert predict("I have a mild cough") == "ROUTINE"
    assert predict("meri tabiyat theek nahi hai") == "INSUFFICIENT"
    assert predict("the benefits are fine") == "INSUFFICIENT"   # 'fits' must not match inside 'benefits'


def _gold():
    rows = []
    for cid, lab, pair in [("C1", "EMERGENCY", "P01"), ("C2", "ROUTINE", "P01"), ("C3", "EMERGENCY", None)]:
        for s in ("en", "deva", "hinglish"):
            rows.append({"id": f"{cid}|{s}", "case_id": cid, "script": s, "text": "x",
                         "label": lab, "pair_id": pair})
    return rows


def test_evaluate_perfect_and_pairs():
    gold = _gold()
    preds = {g["id"]: {"pred": g["label"]} for g in gold}
    r = ev.evaluate(gold, preds, n_boot=50)
    assert r["overall"]["emergency_recall"] == 1.0
    assert r["minimal_pairs"]["en"]["flip_acc"] == 1.0
    assert r["cross_script_recall_gap"] == 0.0


def test_evaluate_all_emergency_fails_flip():
    gold = _gold()
    preds = {g["id"]: {"pred": "EMERGENCY"} for g in gold}
    r = ev.evaluate(gold, preds, n_boot=50)
    assert r["overall"]["emergency_recall"] == 1.0
    assert r["overall"]["over_triage"] == 1.0
    assert r["minimal_pairs"]["deva"]["flip_acc"] == 0.0


def test_missing_prediction_raises():
    gold = _gold()
    with pytest.raises(ValueError):
        ev.evaluate(gold, {}, n_boot=10)
