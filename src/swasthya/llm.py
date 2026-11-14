"""Orchestrator LLM client: an OpenAI-compatible chat endpoint with tool calling, plus a scripted fake.

The client sends temperature 0, retries rate limits and server errors with backoff, and raises LLMError when it
gives up (the orchestrator records that as an infrastructure error so the episode can be rerun, not scored as an
agent failure). The API key is only ever put in the request header; it is never logged or stored in results.
"""
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, List


class LLMError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: Any                    # dict, or None when the model produced arguments that are not a JSON object


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _parse_args(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
        except ValueError:
            return None
        return v if isinstance(v, dict) else None
    return None


class OpenAICompatClient:
    def __init__(self, model, base_url, api_key, temperature=0.0, max_tokens=512, retries=5,
                 min_interval=0.0, timeout=90, post=None, sleep=time.sleep, clock=time.monotonic):
        self.model, self.url = model, base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self.temperature, self.max_tokens, self.retries = temperature, max_tokens, retries
        self.min_interval, self.timeout = min_interval, timeout
        self._post, self._sleep, self._clock, self._last = post or self._http_post, sleep, clock, None

    def _http_post(self, url, headers, body):
        req = urllib.request.Request(url, data=body.encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    def chat(self, messages, tools):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}
        payload = json.dumps(body, ensure_ascii=False)
        for attempt in range(self.retries):
            if self._last is not None and self.min_interval:
                wait = self.min_interval - (self._clock() - self._last)
                if wait > 0:
                    self._sleep(wait)
            self._last = self._clock()
            try:
                status, text = self._post(self.url, headers, payload)
            except (urllib.error.URLError, TimeoutError, OSError):
                status, text = 599, ""
            if status == 200:
                return self._parse(text)
            if status in (408, 409, 429) or status >= 500:
                self._sleep(min(2 ** attempt, 30))
                continue
            raise LLMError(f"HTTP {status}")
        raise LLMError("retries exhausted")

    @staticmethod
    def _parse(text):
        try:
            data = json.loads(text)
            msg = data["choices"][0]["message"]
            calls = [ToolCall(tc.get("id") or f"call_{i}", tc["function"]["name"],
                              _parse_args(tc["function"].get("arguments")))
                     for i, tc in enumerate(msg.get("tool_calls") or [])]
            usage = data.get("usage") or {}
            return LLMReply(msg.get("content") or "", calls, int(usage.get("prompt_tokens") or 0),
                            int(usage.get("completion_tokens") or 0))
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            raise LLMError("malformed response") from e


class ScriptedLLM:
    """Fake LLM for tests and fault injection. `script` items are LLMReply objects, LLMError instances (raised),
    or callables (messages, tools) -> LLMReply. `calls` records the tool names offered on each call."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, messages, tools):
        self.calls.append({"tools": [t["function"]["name"] for t in tools], "n_messages": len(messages)})
        if not self.script:
            raise AssertionError("LLM was called more often than scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(messages, tools) if callable(item) else item
