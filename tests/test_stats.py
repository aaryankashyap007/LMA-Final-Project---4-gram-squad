import math
from pathlib import Path

from swasthya.scenario import load_scenarios
from swasthya.stats import (build_records, clopper_pearson, compare, holm, interaction, mcnemar_exact,
                            primary_family)

ROOT = Path(__file__).resolve().parents[1]
V1 = {s.id: s for s in load_scenarios(ROOT / "data/d4_clinic/scenarios_adversarial_v1.jsonl")}
IDS = sorted(V1)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 1) == 1.0
    assert mcnemar_exact(2, 0) == 0.5 and mcnemar_exact(0, 2) == 0.5
    assert math.isclose(mcnemar_exact(0, 5), 0.0625)
    assert math.isclose(mcnemar_exact(1, 6), 2 * (1 + 7) / 128)       # 2 * P(X <= 1), n = 7


def test_holm_known_values_and_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adj["a"], 0.03) and math.isclose(adj["c"], 0.06) and math.isclose(adj["b"], 0.06)
    assert all(v <= 1 for v in holm({"x": 0.9, "y": 0.8}).values())


def test_clopper_pearson_zero_and_all_and_middle():
    lo, hi = clopper_pearson(0, 90)
    assert lo == 0.0 and math.isclose(hi, 1 - 0.025 ** (1 / 90), abs_tol=1e-4)
    lo, hi = clopper_pearson(90, 90)
    assert hi == 1.0 and math.isclose(lo, 0.025 ** (1 / 90), abs_tol=1e-4)
    lo, hi = clopper_pearson(5, 10)
    assert 0.18 < lo < 0.20 and 0.80 < hi < 0.82               # known interval 0.187-0.813
    assert clopper_pearson(0, 0) == (0.0, 1.0)


def episode(sid, arm, harm=False, flags=None):
    return {"scenario_id": sid, "arm": arm, "audit": [], "flags": flags or {},
            "score": {"unapproved_actions": int(harm), "leaks": 0, "task_success": True}}


def records(harmed):
    """harmed: {(scenario id, arm)} -> episodes for the four arms over every v1 attack scenario."""
    eps = [episode(sid, arm, (sid, arm) in harmed) for sid in IDS for arm in ("A1", "+Gate", "+Guard", "Full")]
    return build_records([("r", eps, V1)])


def test_a_guard_effect_is_found_and_the_gate_effect_is_zero():
    sids = IDS[:12]
    harmed = {(s, a) for s in sids for a in ("A1", "+Gate")}                 # the guard-off arms are harmed
    recs, excluded = records(harmed)
    assert excluded == {}
    fam = primary_family(recs, reps=500)
    g = fam["guard effect, gate off"]
    assert (g["only_a"], g["only_b"], g["diff"]) == (12, 0, 12 / len(IDS))
    assert math.isclose(g["p"], mcnemar_exact(12, 0)) and g["p"] < 0.001
    assert g["ci"][0] > 0 and g["p_holm"] >= g["p"]
    assert fam["gate effect, guard off"]["diff"] == 0 and fam["gate effect, guard off"]["p"] == 1.0
    ix = interaction(recs, reps=500)
    assert math.isclose(ix["value"], 0.0, abs_tol=1e-12)                       # same guard effect with and without gate


def test_an_interaction_is_detected():
    sids = IDS[:12]
    harmed = {(s, "A1") for s in sids}                                         # only the plain baseline is harmed
    recs, _ = records(harmed)
    ix = interaction(recs, reps=500)
    assert math.isclose(ix["value"], 12 / len(IDS)) and ix["ci"][0] >= 0


def test_family_level_test_is_more_conservative_than_scenario_level():
    sids = [s for s in IDS if s.startswith("rel_cancel")]                      # 3 scripts of one family
    recs, _ = records({(s, "A1") for s in sids})
    g = compare(recs, "A1", "+Guard", "harm", "attack", 200)
    assert g["only_a"] == 3 and g["fam_only_a"] == 1 and g["p_family"] > g["p"]


def test_flagged_episodes_are_excluded_not_counted_as_safe():
    eps = [episode(IDS[0], "A1"), episode(IDS[0], "+Guard", flags={"infra_error": True}),
           episode(IDS[1], "A1", flags={"gate_outage": True})]
    recs, excluded = build_records([("r", eps, V1)])
    assert excluded == {"r": 2}
    assert list(recs[("r", IDS[0])]["arms"]) == ["A1"] and ("r", IDS[1]) not in recs
    assert compare(recs, "A1", "+Guard", "harm", "attack", 50) is None        # no complete pair


def test_bootstrap_is_reproducible():
    recs, _ = records({(s, "A1") for s in IDS[:5]})
    assert compare(recs, "A1", "+Guard", reps=300, seed=7)["ci"] == compare(recs, "A1", "+Guard", reps=300, seed=7)["ci"]


from swasthya.stats import call_metrics, clinical_turn, escalation_delay, handoff_complete, noninferiority  # noqa: E402

BENIGN = {s.id: s for s in load_scenarios(ROOT / "data/d4_clinic/scenarios_benign_v0.jsonl")}


def test_call_metrics_counts_schema_blocked_and_matched_calls():
    sc = BENIGN["book_rao_en"]                          # authorises one booking
    book = sc.gold.authorized[0]
    audit = [{"tool": "book_appointment", "args": book["args"], "mutating": True, "executed": False, "ok": False,
              "verdict": "needs_confirmation"},
             {"tool": "lookup_policy", "args": {}, "mutating": False, "executed": True, "ok": False,
              "verdict": "schema_invalid"},
             {"tool": "book_appointment", "args": book["args"], "mutating": True, "executed": True, "ok": True,
              "verdict": "ok"},
             {"tool": "cancel_appointment", "args": {"appointment_id": "A001"}, "mutating": True, "executed": True,
              "ok": True, "verdict": "ok"}]
    m = call_metrics({"audit": audit}, sc)
    assert m == {"calls": 4, "schema_invalid": 1, "restricted": 0, "mutating_ok": 2, "matched": 1, "gold_n": 1}


def test_clinical_turn_and_escalation_delay():
    sc = BENIGN["emergency_interrupts_booking_en"]
    assert clinical_turn(sc) == 2                        # the chest pain arrives on the second message
    assert escalation_delay({"escalation_turn": 2}, sc) == 0
    assert escalation_delay({"escalation_turn": 3}, sc) == 1
    assert escalation_delay({"escalation_turn": None}, sc) is None
    assert clinical_turn(BENIGN["book_rao_en"]) == 1


def test_handoff_completeness():
    full = {"urgency": "EMERGENCY", "source": "gate", "script": "en", "patient_text_untrusted": "x", "patient_id": "P004"}
    assert handoff_complete(full, True) == 1.0
    assert handoff_complete({**full, "patient_id": None}, True) == 0.8
    assert handoff_complete({**full, "patient_id": None}, False) == 1.0       # unverified caller: no id expected


def test_noninferiority_verdict_depends_on_the_margin():
    sids = [s for s in BENIGN if s.endswith("_en")]
    eps = []
    for i, sid in enumerate(sids):
        for arm in ("A1", "Full"):
            eps.append({"scenario_id": sid, "arm": arm, "audit": [], "flags": {},
                        "score": {"unapproved_actions": 0, "leaks": 0, "task_success": not (arm == "Full" and i < 3)}})
    recs, _ = build_records([("b", eps, BENIGN)])
    loose = noninferiority(recs, margin=0.50, reps=300)
    tight = noninferiority(recs, margin=0.01, reps=300)
    assert loose["noninferior"] and not tight["noninferior"] and loose["diff"] < 0
