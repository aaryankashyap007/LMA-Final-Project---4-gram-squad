"""Gate wrapper for the agent: caches per-message class probabilities and fails closed.

The backend (G1 = Qwen2.5-3B + QLoRA adapter, or V0 = the same model prompted) is anything with
`probs(texts) -> [[p_E, p_U, p_R, p_I], ...]`, ordered as schema.LABELS. Probabilities are cached on disk keyed by
(checkpoint id, normalised text), so scripted scenarios are computed once (e.g. in a single Kaggle batch) and
every rerun, on any machine, is free and repeatable. Changing the decision rule (argmax, or an emergency
threshold chosen on dev) needs no recomputation because probabilities, not labels, are cached.

Fail closed: if the backend is missing or raises, the result is UNAVAILABLE and the orchestrator treats the turn
as unsafe (hand-off) and flags the episode as a gate outage. Failures are never cached.
"""
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import prompts
from .normaliser import for_model
from .schema import LABELS

UNAVAILABLE = "UNAVAILABLE"


def checkpoint_id_from_dir(adapter_dir, name="g1"):
    """Cache id derived from the adapter's actual files, so a different adapter can never reuse old probabilities."""
    h = hashlib.sha256()
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        p = Path(adapter_dir) / fn
        if not p.exists():
            raise FileNotFoundError(p)
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return f"{name}-{h.hexdigest()[:12]}"


def prompted_checkpoint_id(model_name, shots=(), name="v0"):
    """Cache id for the prompted gate: model, rubric prompt and few-shot examples all change the probabilities."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(prompts.SYSTEM.encode("utf-8"))
    h.update(json.dumps([[s["id"], s["text"], s["label"]] for s in shots], ensure_ascii=False).encode("utf-8"))
    return f"{name}-{h.hexdigest()[:12]}"


def normalise(text):
    """The conservative model-facing normalisation (normaliser.for_model) applied before the gate and before cache
    lookup. It keeps line breaks, which separate the speakers in the transcript input the probe validated."""
    return for_model(text)


@dataclass
class GateResult:
    label: str                          # one of schema.LABELS, or UNAVAILABLE
    probs: Optional[List[float]] = None
    source: str = "live"                # cache | live | fail_closed

    @property
    def outage(self):
        return self.label == UNAVAILABLE


def decide(probs, emergency_threshold=None):
    """Most likely class, unless P(EMERGENCY) reaches the threshold (set on dev, as in the H1 analysis)."""
    if emergency_threshold is not None and probs[0] >= emergency_threshold:
        return "EMERGENCY"
    return LABELS[max(range(len(probs)), key=probs.__getitem__)]


def _valid(p):
    return (isinstance(p, (list, tuple)) and len(p) == len(LABELS)
            and all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in p)
            and abs(sum(p) - 1.0) < 1e-3)


class GateCache:
    """Append-only jsonl: one {"key", "text", "probs"} per line. A torn last line is ignored."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.data = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.data[r["key"]] = r["probs"]
                except (ValueError, KeyError):
                    continue

    def get(self, key):
        return self.data.get(key)

    def put(self, key, text, probs):
        self.data[key] = probs
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "text": text, "probs": probs}, ensure_ascii=False) + "\n")


class CachedGate:
    def __init__(self, backend, checkpoint_id, cache_path=None, emergency_threshold=None):
        self.backend = backend                       # None = cache only (every miss fails closed)
        self.checkpoint_id = checkpoint_id           # e.g. "g1-adapter-s13": a new checkpoint is a new cache
        self.cache = GateCache(cache_path)
        self.emergency_threshold = emergency_threshold
        self.stats = {"cache": 0, "live": 0, "fail_closed": 0}

    def key(self, text):
        return hashlib.sha256(f"{self.checkpoint_id}\x00{normalise(text)}".encode("utf-8")).hexdigest()

    def predict(self, text):
        return self.predict_many([text])[0]

    def predict_many(self, texts):
        keys = [self.key(t) for t in texts]
        missing, seen = [], set()
        for t, k in zip(texts, keys):
            if self.cache.get(k) is None and k not in seen:
                seen.add(k)
                missing.append((k, normalise(t)))
        fresh = set()
        if missing and self.backend is not None:
            try:
                out = self.backend.probs([t for _, t in missing])
                for (k, t), p in zip(missing, out):
                    if _valid(p):
                        self.cache.put(k, t, [float(x) for x in p])
                        fresh.add(k)
            except Exception:                        # any backend failure: fail closed, never cache it
                pass
        results = []
        for k in keys:
            p = self.cache.get(k)
            if p is None:
                self.stats["fail_closed"] += 1
                results.append(GateResult(UNAVAILABLE, None, "fail_closed"))
            else:
                src = "live" if k in fresh else "cache"
                self.stats[src] += 1
                results.append(GateResult(decide(p, self.emergency_threshold), p, src))
        return results

    def warm(self, texts):
        """Fill the cache for many texts (e.g. all scripted user turns, in one GPU session)."""
        before = len(self.cache.data)
        self.predict_many(list(texts))
        return len(self.cache.data) - before
