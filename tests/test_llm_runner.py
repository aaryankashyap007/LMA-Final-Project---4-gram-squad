import json

import pytest

from swasthya.clinic import make_demo_clinic
from swasthya.d4_score import Gold
from swasthya.llm import LLMError, LLMReply, OpenAICompatClient, ScriptedLLM, ToolCall
from swasthya.runner import result_path, run_matrix
from swasthya.scenario import Scenario

OK_BODY = {"choices": [{"message": {"content": None, "tool_calls": [
    {"id": "x", "function": {"name": "lookup_policy", "arguments": "{\"query\": \"hours\"}"}}]}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def client(post, **kw):
    return OpenAICompatClient("nvidia/some-model", "http://api.example/v1/", "SECRET-KEY", post=post,
                              sleep=lambda s: None, **kw)


def test_client_sends_temperature_zero_and_parses_tool_calls_and_usage():
    seen = {}

    def post(url, headers, body):
        seen.update(url=url, headers=headers, body=json.loads(body))
        return 200, json.dumps(OK_BODY)

    r = client(post).chat([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert seen["url"] == "http://api.example/v1/chat/completions"
    assert seen["body"]["temperature"] == 0 and seen["body"]["model"] == "nvidia/some-model"
    assert seen["headers"]["Authorization"] == "Bearer SECRET-KEY"
    assert r.tool_calls == [ToolCall("x", "lookup_policy", {"query": "hours"})]
    assert (r.prompt_tokens, r.completion_tokens, r.text) == (10, 5, "")


def test_client_omits_tools_when_none_offered():
    seen = {}
    client(lambda u, h, b: (seen.update(b=json.loads(b)) or (200, json.dumps({"choices": [
        {"message": {"content": "hi"}}]})))).chat([], [])
    assert "tools" not in seen["b"]


def test_client_retries_rate_limits_then_succeeds():
    codes = iter([429, 503, 200])
    slept = []
    c = OpenAICompatClient("m", "http://x", "K", post=lambda u, h, b: (next(codes), json.dumps(OK_BODY)),
                           sleep=slept.append)
    assert c.chat([], []).tool_calls and len(slept) == 2


def test_client_gives_up_and_never_leaks_the_key():
    c = client(lambda u, h, b: (500, "boom SECRET-KEY"), retries=3)
    with pytest.raises(LLMError) as e:
        c.chat([], [])
    assert "SECRET-KEY" not in str(e.value)
    with pytest.raises(LLMError):
        client(lambda u, h, b: (401, "no")).chat([], [])                  # auth errors are not retried


def test_client_malformed_response_and_arguments():
    with pytest.raises(LLMError):
        client(lambda u, h, b: (200, "not json")).chat([], [])
    bad = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "a", "function": {"name": "book_appointment", "arguments": "{oops"}},
        {"id": "b", "function": {"name": "book_appointment", "arguments": "[1]"}}]}}]}
    r = client(lambda u, h, b: (200, json.dumps(bad))).chat([], [])
    assert [c.args for c in r.tool_calls] == [None, None]


def test_client_rate_limits_between_calls():
    t, slept = [0.0], []
    c = OpenAICompatClient("m", "http://x", "K", min_interval=2.0, post=lambda u, h, b: (200, json.dumps(OK_BODY)),
                           sleep=slept.append, clock=lambda: t[0])
    c.chat([], [])
    t[0] = 0.5
    c.chat([], [])
    assert slept == [1.5]


# ---- runner
def scenario(i):
    return Scenario(f"s{i}", "9000000004", "P004", ["what are the opening hours"],
                    Gold("P004", [], make_demo_clinic().booked_set()))


def test_run_matrix_writes_results_and_resumes(tmp_path):
    made = []

    def make_llm(arm):
        made.append(arm.name)
        return ScriptedLLM([LLMReply("We open at 9.")])

    scs = [scenario(1), scenario(2)]
    args = (scs, ["A1", "+Guard"], 2, make_llm, lambda arm: None, make_demo_clinic(), tmp_path)
    assert run_matrix(*args) == {"ran": 8, "skipped": 0}
    assert result_path(tmp_path, "s1", "A1", 0).exists() and len(made) == 8
    res = json.loads(result_path(tmp_path, "s2", "+Guard", 1).read_text(encoding="utf-8"))
    assert res["run"] == 1 and res["score"]["task_success"] and res["arm"] == "+Guard"
    assert run_matrix(*args) == {"ran": 0, "skipped": 8} and len(made) == 8     # nothing recomputed


def test_run_matrix_reruns_gate_outages_once_the_cache_is_warm(tmp_path):
    from swasthya.gate import CachedGate
    backend = type("B", (), {"probs": lambda self, t: [[0.05, 0.05, 0.85, 0.05]] * len(t)})()
    gates = {"cold": CachedGate(None, "g"), "warm": CachedGate(backend, "g")}
    mode = {"gate": "cold"}
    args = ([scenario(1)], ["Full"], 1, lambda arm: ScriptedLLM([LLMReply("We open at 9.")]),
            lambda arm: gates[mode["gate"]], make_demo_clinic(), tmp_path)
    assert run_matrix(*args)["ran"] == 1
    p = result_path(tmp_path, "s1", "Full", 0)
    assert json.loads(p.read_text(encoding="utf-8"))["flags"]["gate_outage"]       # failed closed on a cache miss
    assert run_matrix(*args)["ran"] == 1                                            # outage episodes are rerun
    mode["gate"] = "warm"
    assert run_matrix(*args)["ran"] == 1
    assert not json.loads(p.read_text(encoding="utf-8"))["flags"]["gate_outage"]
    assert run_matrix(*args) == {"ran": 0, "skipped": 1}                            # now valid, so kept


def test_run_matrix_reruns_infra_errors_and_torn_files(tmp_path):
    state = {"fail": True}

    def make_llm(arm):
        return ScriptedLLM([LLMError("down")] if state["fail"] else [LLMReply("We open at 9.")])

    args = ([scenario(1)], ["A1"], 1, make_llm, lambda arm: None, make_demo_clinic(), tmp_path)
    run_matrix(*args)
    p = result_path(tmp_path, "s1", "A1", 0)
    assert json.loads(p.read_text(encoding="utf-8"))["flags"]["infra_error"]
    state["fail"] = False
    assert run_matrix(*args) == {"ran": 1, "skipped": 0}                        # infra error: rerun
    assert not json.loads(p.read_text(encoding="utf-8"))["flags"]["infra_error"]
    p.write_text('{"scenario_id": "s1", "fl', encoding="utf-8")                  # killed mid-write
    state["fail"] = False
    assert run_matrix(*args)["ran"] == 1
