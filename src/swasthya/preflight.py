"""Quick checks that the API behaves as the experiment assumes, before spending the quota on a full run.

Hard checks (the run cannot work without them): the endpoint answers, it makes a well-formed tool call in English,
and it makes one in Hindi. Soft checks (worth knowing): temperature 0 repeats the same text, and token usage is
reported (otherwise the token budget falls back to a character-count estimate).
"""
import time

from .llm import LLMError
from .orchestrator import tool_schema
from .tools import validate_args

PING = [{"role": "user", "content": "Reply with the single word OK."}]
_SYS = {"role": "system", "content": "You are a clinic front-desk assistant. Always use a tool when one applies."}
EN = [_SYS, {"role": "user", "content": "What are the clinic's opening hours? Use the lookup_policy tool."}]
HI = [_SYS, {"role": "user", "content": "क्लिनिक कब खुलता है? lookup_policy टूल का इस्तेमाल करें।"}]


def _tool_call_ok(client, messages):
    reply = client.chat(messages, [tool_schema("lookup_policy")])
    if not reply.tool_calls:
        return False, f"no tool call (text: {reply.text[:80]!r})"
    c = reply.tool_calls[0]
    errs = validate_args(c.name, c.args)
    if c.name != "lookup_policy" or errs:
        return False, f"bad call {c.name} {c.args!r} {errs}"
    return True, f"{c.name}({c.args})"


def run_preflight(client):
    """Returns {"ok": bool, "checks": [{name, hard, ok, detail, seconds}]}. ok = every hard check passed."""
    checks = []

    def check(name, hard, fn):
        t0 = time.time()
        try:
            ok, detail = fn()
        except LLMError as e:
            ok, detail = False, f"API error: {e}"
        checks.append({"name": name, "hard": hard, "ok": bool(ok), "detail": detail,
                       "seconds": round(time.time() - t0, 2)})

    def plain():
        r = client.chat(PING, [])
        return bool(r.text.strip()), f"reply {r.text.strip()[:40]!r}"

    def usage():
        r = client.chat(PING, [])
        return r.prompt_tokens > 0 and r.completion_tokens > 0, \
            f"prompt {r.prompt_tokens}, completion {r.completion_tokens}"

    def repeatable():
        a, b = client.chat(PING, []).text, client.chat(PING, []).text
        return a == b, "identical" if a == b else f"differs: {a[:30]!r} vs {b[:30]!r}"

    check("answers", True, plain)
    check("tool_call_english", True, lambda: _tool_call_ok(client, EN))
    check("tool_call_hindi", True, lambda: _tool_call_ok(client, HI))
    check("usage_reported", False, usage)
    check("temperature_zero_repeatable", False, repeatable)
    return {"ok": all(c["ok"] for c in checks if c["hard"]), "checks": checks}
