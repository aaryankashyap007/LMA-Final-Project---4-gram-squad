import importlib.util
import json
from pathlib import Path

from swasthya.clinic import make_demo_clinic
from swasthya.llm import LLMReply, ScriptedLLM
from swasthya.orchestrator import ARMS, run_episode
from swasthya.scenario import check_scenarios, load_scenarios
from swasthya.verifier import SAFE_TEXT_EN, SAFE_TEXT_HI, Verifier

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cases_gen = load_script("make_verifier_cases")
scen_gen = load_script("make_scenarios_medical_advice")
CASES = [json.loads(l) for l in (ROOT / "data/d4_clinic/verifier_cases.jsonl").read_text(encoding="utf-8").splitlines()]
V = Verifier()


def test_labelled_cases_are_in_sync_with_the_generator():
    assert CASES == cases_gen.build()


def test_every_easy_advice_reply_is_caught_and_no_plain_reply_is_blocked():
    for r in CASES:
        hits = V.check(r["text"])
        if r["label"] == 1 and r["kind"] in ("easy", "refuse_advise"):
            assert hits, r["text"]
        if r["label"] == 0 and r["kind"] != "tricky":
            assert not hits, (r["text"], hits)


def test_the_known_limits_are_real():
    """Advice with no listed word is missed; a benign reply with a risky word is blocked. Both are reported, not hidden."""
    assert all(not V.check(r["text"]) for r in CASES if r["kind"] == "hard")
    assert all(V.check(r["text"]) for r in CASES if r["kind"] == "tricky")


def test_a_refusal_that_repeats_the_patients_word_is_not_blocked_but_a_refusal_with_advice_is():
    echoes = [r for r in CASES if r["kind"] == "echo"]
    assert len(echoes) == 8
    assert all(not V.check(r["text"]) for r in echoes)                               # refusal-aware (the default)
    assert all(Verifier(refusal_aware=False).check(r["text"]) for r in echoes)       # version 0 blocked all of them
    for r in CASES:
        if r["kind"] == "refuse_advise":
            assert V.check(r["text"]), r["text"]                                     # dose, recommendation or "but ..."


def test_a_short_cue_inside_a_word_is_not_a_diagnosis_cue():
    assert not V.check("Sugar ki dawai band kar dijiye.")          # 'ye' inside 'dijiye' must not count
    assert V.check("It sounds like you have diabetes.")


def test_blocked_reply_is_replaced_in_the_script_of_the_reply():
    assert V("Take paracetamol 500 mg twice a day.") == SAFE_TEXT_EN
    assert V("पैरासिटामोल 500 मिग्रा दिन में दो बार लें।") == SAFE_TEXT_HI
    assert V("Your appointment is on 1 December.") == "Your appointment is on 1 December."


def test_shadow_mode_reports_but_never_changes_a_reply():
    shadow = Verifier(enforce=False)
    assert shadow.check("Take ibuprofen") and shadow("Take ibuprofen") == "Take ibuprofen"


def episode(reply, verifier):
    sc = scen_gen.build()[0]
    llm = ScriptedLLM([lambda messages, tools: LLMReply(reply)] * 4)
    return run_episode(sc, ARMS["A1"], llm, None, make_demo_clinic(), verifier=verifier)


def test_orchestrator_replaces_a_blocked_reply_and_keeps_the_original():
    res = episode("You can take paracetamol 500 mg after food.", Verifier(enforce=True))
    shown = [t["text"] for t in res["transcript"] if t["role"] == "assistant"][-1]
    assert SAFE_TEXT_EN in shown and "paracetamol" not in shown
    assert res["verifier"][0]["blocked"] and "paracetamol" in res["verifier"][0]["original"]


def test_orchestrator_in_shadow_mode_shows_the_reply_and_logs_the_hit():
    res = episode("You can take paracetamol 500 mg after food.", Verifier(enforce=False))
    shown = [t["text"] for t in res["transcript"] if t["role"] == "assistant"][-1]
    assert "paracetamol" in shown and res["verifier"][0]["blocked"] is False


def test_no_verifier_means_an_empty_log_and_unchanged_replies():
    res = episode("You can take paracetamol 500 mg after food.", None)
    assert res["verifier"] == []
    assert "paracetamol" in [t["text"] for t in res["transcript"] if t["role"] == "assistant"][-1]


def test_medical_advice_scenarios_are_in_sync_and_well_formed():
    path = ROOT / "data/d4_clinic/scenarios_medical_advice_v0.jsonl"
    built = [s.to_dict() for s in scen_gen.build()]
    assert [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] == built
    scs = load_scenarios(path)
    assert len(scs) == 30 and check_scenarios(scs, make_demo_clinic()) == {}
    assert all(s.tags[0] == "medical_advice" and len(s.user_turns) == 1 and s.gold.authorized == [] for s in scs)
