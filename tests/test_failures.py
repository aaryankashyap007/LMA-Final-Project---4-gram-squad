from swasthya.failures import error_type, join, pick_candidates, summarise
from swasthya.schema import LABELS, SCRIPTS


def test_error_types():
    cases = {("EMERGENCY", "ROUTINE"): "missed_emergency", ("EMERGENCY", "INSUFFICIENT"): "missed_emergency",
             ("ROUTINE", "EMERGENCY"): "false_alarm", ("URGENT", "EMERGENCY"): "false_alarm",
             ("INSUFFICIENT", "URGENT"): "committed_on_vague", ("URGENT", "INSUFFICIENT"): "wrongly_insufficient",
             ("URGENT", "ROUTINE"): "under_triage", ("ROUTINE", "URGENT"): "over_triage",
             ("URGENT", "URGENT"): "correct"}
    for (g, p), want in cases.items():
        assert error_type(g, p) == want, (g, p)


def probs_for(label, conf=0.8):
    rest = (1 - conf) / 3
    return [conf if l == label else rest for l in LABELS]


def build(spec):
    """spec: {case_id: (gold, {script: pred}, pair_id)} -> (gold rows, preds dict)."""
    gold, preds = [], {}
    for cid, (lab, per_script, pair) in spec.items():
        for s in SCRIPTS:
            rid = f"{cid}|{s}"
            gold.append({"id": rid, "case_id": cid, "script": s, "label": lab, "category": "heart_brain",
                         "pair_id": pair, "text": f"text {rid}", "cues": ["x"]})
            preds[rid] = {"id": rid, "pred": per_script[s], "probs": probs_for(per_script[s])}
    return gold, preds


SPEC = {
    "C1": ("EMERGENCY", {"en": "EMERGENCY", "deva": "ROUTINE", "hinglish": "EMERGENCY"}, None),    # script-specific
    "C2": ("EMERGENCY", {s: "URGENT" for s in SCRIPTS}, "P1"),                                    # pair, all wrong
    "C3": ("URGENT", {s: "URGENT" for s in SCRIPTS}, "P1"),
    "C4": ("ROUTINE", {"en": "ROUTINE", "deva": "ROUTINE", "hinglish": "EMERGENCY"}, None),       # false alarm
    "C5": ("URGENT", {s: "ROUTINE" for s in SCRIPTS}, None),                                      # under-triage
    "C6": ("INSUFFICIENT", {s: "INSUFFICIENT" for s in SCRIPTS}, None),
}


def rows():
    g, p = build(SPEC)
    return join(g, p)


def test_summary_counts():
    s = summarise(rows())
    assert s["n"] == 18 and s["n_errors"] == 1 + 3 + 1 + 3
    assert s["by_type"]["missed_emergency"] == 4 and s["by_type"]["false_alarm"] == 1
    assert s["by_type"]["under_triage"] == 3
    assert s["by_script"] == {"en": 2, "deva": 3, "hinglish": 3}
    assert s["cases_by_n_wrong_scripts"] == {0: 2, 1: 2, 3: 2}
    assert s["confusion"]["EMERGENCY"]["URGENT"] == 3
    assert {p[:2] for p in s["pair_failures"]} == {("P1", sc) for sc in SCRIPTS}   # both members predicted URGENT


def test_pair_failures_only_when_both_members_get_the_same_prediction():
    g, p = build({"C2": ("EMERGENCY", {s: "URGENT" for s in SCRIPTS}, "P1"),
                  "C3": ("URGENT", {s: "URGENT" for s in SCRIPTS}, "P1")})
    pf = summarise(join(g, p))["pair_failures"]
    assert len(pf) == 3 and all(x[0] == "P1" and x[4] == "URGENT" for x in pf)
    g, p = build({"C2": ("EMERGENCY", {s: "EMERGENCY" for s in SCRIPTS}, "P1"),
                  "C3": ("URGENT", {s: "URGENT" for s in SCRIPTS}, "P1")})
    assert summarise(join(g, p))["pair_failures"] == []


def test_candidates_cover_failure_modes_are_unique_and_bounded():
    picked = pick_candidates(rows(), k=8)
    ids = [r["id"] for _, r in picked]
    assert len(ids) == len(set(ids)) and len(picked) <= 8
    reasons = [why for why, _ in picked]
    for mode in ("missed emergency", "false alarm", "script-specific failure", "minimal pair did not flip",
                 "under-triage URGENT->ROUTINE"):
        assert any(mode in w for w in reasons), mode          # every mode covered, possibly sharing a row
    assert all(r["etype"] != "correct" for _, r in picked)


def test_candidates_are_deterministic_and_respect_k():
    assert [r["id"] for _, r in pick_candidates(rows(), 8)] == [r["id"] for _, r in pick_candidates(rows(), 8)]
    assert len(pick_candidates(rows(), 3)) == 3


def test_no_errors_gives_no_candidates():
    g, p = build({"C6": ("INSUFFICIENT", {s: "INSUFFICIENT" for s in SCRIPTS}, None)})
    assert pick_candidates(join(g, p)) == []
