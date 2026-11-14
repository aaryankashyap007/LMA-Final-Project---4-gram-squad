import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from swasthya.clinic import make_demo_clinic
from swasthya.guard import is_affirmation
from swasthya.llm import LLMReply, ScriptedLLM, ToolCall
from swasthya.orchestrator import ARMS, run_episode
from swasthya.scenario import check_scenarios, load_scenarios

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/d4_clinic/scenarios_benign_v0.jsonl"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load_script("make_scenarios_benign")
probe = load_script("probe_context")
DEVA = re.compile("[ऀ-ॿ]")


# ---- the benign scenario set
def test_committed_file_is_in_sync_with_the_generator_and_valid():
    built = [s.to_dict() for s in gen.build()]
    on_disk = [json.loads(l) for l in FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert on_disk == built, "run: python scripts/make_scenarios_benign.py"
    assert check_scenarios(load_scenarios(FILE), make_demo_clinic()) == {}


def test_every_family_has_all_three_scripts_with_matching_structure():
    sc = load_scenarios(FILE)
    assert len(sc) == 39 and len({s.id for s in sc}) == 39
    by_family = {}
    for s in sc:
        by_family.setdefault(s.tags[1], []).append(s)
    assert len(by_family) == 13
    for fam, group in by_family.items():
        assert {s.tags[2] for s in group} == {"en", "hinglish", "deva"}, fam
        assert len({len(s.user_turns) for s in group}) == 1, fam               # same number of turns in each script
        assert len({json.dumps(s.gold.authorized, sort_keys=True) for s in group}) == 1, fam
        assert len({tuple(sorted(s.gold.final_booked)) for s in group}) == 1, fam


def test_scripts_use_the_right_alphabet():
    for s in load_scenarios(FILE):
        has_deva = [bool(DEVA.search(t)) for t in s.user_turns]
        script = s.tags[2]
        assert all(has_deva) if script == "deva" else not any(has_deva), s.id


def test_set_covers_the_intended_situations():
    sc = {s.tags[1]: s for s in load_scenarios(FILE) if s.tags[2] == "en"}
    assert sc["proxy_book"].caller_phone != make_demo_clinic().get_patient("P002").phone    # a registered proxy
    assert sc["proxy_book"].gold.patient_id == "P002"
    assert sc["clinic_hours"].gold.patient_id is None and sc["clinic_hours"].claimed_patient_id is None
    assert len(sc["reschedule"].gold.authorized) == 2 and len(sc["reschedule"].user_turns) == 3
    for fam in ("emergency_interrupts_booking", "urgent_child", "medical_advice", "info_my_appointment",
                "policy_question", "clinic_hours"):
        assert sc[fam].gold.authorized == [], fam                                # nothing may change


def perfect_agent(scenario):
    """An LLM that does exactly what the scenario's gold authorises, one confirmation at a time."""
    remaining = [(a["tool"], a["args"]) for a in scenario.gold.authorized]
    greetings = {"Hello", "Namaste", "नमस्ते"}

    def call():
        tool, args = remaining[0]
        return LLMReply("", [ToolCall("c", tool, args)])

    def policy(messages, tools):
        last = messages[-1]
        if last["role"] == "tool":
            if last["content"].startswith("Confirmation required"):
                return LLMReply("Please confirm.")
            if last["content"].startswith("{"):                                   # the action went through
                remaining.pop(0)
                return call() if remaining else LLMReply("Done.")
            return LLMReply("Sorry, that did not work.")
        text = last["content"]
        if text in greetings or not remaining:
            return LLMReply("How can I help?")
        return call()

    return ScriptedLLM([policy] * 40)


@pytest.mark.parametrize("arm", ["+Guard", "A1"])
def test_every_benign_scenario_is_solvable_through_the_confirmation_protocol(arm):
    """If a real run fails a scenario, the model is at fault, not the script: a perfect agent passes all 39, in the
    guard arm with no automatic confirmations needed (the scripted 'yes' replies are accepted by the guard)."""
    for s in load_scenarios(FILE):
        res = run_episode(s, ARMS[arm], perfect_agent(s), None, make_demo_clinic())
        assert res["score"]["task_success"], (arm, s.id, res["audit"])
        assert res["score"]["unapproved_actions"] == 0 and res["score"]["leaks"] == 0, (arm, s.id)
        assert res["auto_confirmations"] == 0, (arm, s.id)


def test_scripted_confirmation_turns_are_accepted_by_the_guard():
    yes = {t for s in load_scenarios(FILE) if s.gold.authorized
           for t in s.user_turns[2 if "greeting_first" in s.tags else 1:]}      # the turns after the request
    assert yes and all(is_affirmation(t) for t in yes), [t for t in yes if not is_affirmation(t)]


# ---- the administrative probe group
def test_admin_probe_group_is_balanced_and_complete():
    dev = [{"id": f"C{i:03d}|{s}", "case_id": f"C{i:03d}", "script": s, "text": f"t{i}", "label": "ROUTINE" if i < 3 else "URGENT",
            "category": "x"} for i in range(6) for s in ("en", "deva", "hinglish")]
    items = probe.build_items(dev)
    admin = [i for i in items if i["group"] == "admin"]
    assert len(admin) == 42
    for s in ("en", "hinglish", "deva"):
        mine = [i for i in admin if i["script"] == s]
        assert len(mine) == 14 and {i["kind"] for i in mine} == {
            "book", "cancel", "reschedule", "my_appointments", "clinic_info", "greeting"}
    assert all(i["gold"] == "ROUTINE" for i in admin if i["kind"] != "greeting")
    assert all(i["gold"] is None for i in admin if i["kind"] == "greeting")
    assert all(bool(DEVA.search(i["format1"])) == (i["script"] == "deva") for i in admin)
    f3 = next(i for i in admin if i["script"] == "en")["format3"].split("\n")
    assert f3[0].startswith("Patient: ") and f3[1].startswith("Assistant: ") and f3[2].startswith("Patient: ")
    assert len({i["id"] for i in items}) == len(items)


def test_wilson_interval():
    lo, hi = probe.wilson(0, 39)
    assert lo == 0 or lo < 1e-9
    assert 0.0 < hi < 0.12
    lo, hi = probe.wilson(20, 40)
    assert 0.34 < lo < 0.36 and 0.64 < hi < 0.66
    assert probe.wilson(0, 0)[0] != probe.wilson(0, 0)[0]                       # NaN for no data


def test_admin_summary_counts_harmful_labels_and_ignores_greetings():
    items = [{"group": "admin", "kind": k, "script": s, "id": f"a{i}"} for i, (k, s) in enumerate([
        ("book", "en"), ("cancel", "en"), ("clinic_info", "deva"), ("greeting", "en"), ("book", "deva")])]
    preds = {"format1": ["ROUTINE", "INSUFFICIENT", "URGENT", "INSUFFICIENT", "ROUTINE"]}
    s = probe.admin_summary(items, preds, "format1")
    assert s["n"] == 4 and s["harmful"] == 2 and s["harmful_rate"] == 0.5          # the greeting is not counted
    assert s["by_kind"]["cancel"] == [1, 1] and s["by_script"]["deva"] == [1, 2]
    assert s["harmful_ids"] == ["a1", "a2"] and s["labels"]["INSUFFICIENT"] == 2


def test_probe_score_prints_the_admin_section(tmp_path):
    dev = [{"id": f"C{i:03d}|{s}", "case_id": f"C{i:03d}", "script": s, "text": f"t{i}",
            "label": "ROUTINE" if i < 3 else "EMERGENCY", "category": "x"} for i in range(6) for s in ("en", "deva", "hinglish")]
    items = probe.build_items(dev)
    labels = [i["gold"] or "INSUFFICIENT" for i in items]
    (tmp_path / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "preds.json").write_text(json.dumps({f: labels for f in probe.FORMATS}), encoding="utf-8")
    r = subprocess.run([sys.executable, "scripts/probe_context.py", "score", "--items", str(tmp_path / "items.json"),
                        "--preds", str(tmp_path / "preds.json"), "--out", str(tmp_path / "res.json"), "--boot", "20"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "administrative messages" in r.stdout and "harmful 0/39" in r.stdout
    res = json.loads((tmp_path / "res.json").read_text(encoding="utf-8"))
    assert res["admin_format1"]["harmful_rate"] == 0 and len(res["admin_format3"]["by_kind"]) == 5
