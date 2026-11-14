import json

from swasthya.gate import CachedGate, GateCache, UNAVAILABLE, decide, normalise


class FakeBackend:
    def __init__(self, table=None, default=(0.05, 0.05, 0.85, 0.05)):
        self.table, self.default, self.calls = table or {}, list(default), []

    def probs(self, texts):
        self.calls.append(list(texts))
        return [self.table.get(t, self.default) for t in texts]


class BrokenBackend:
    def probs(self, texts):
        raise RuntimeError("gpu on fire")


EMERG = [0.9, 0.05, 0.03, 0.02]


def test_decide_argmax_and_threshold():
    assert decide([0.1, 0.2, 0.6, 0.1]) == "ROUTINE"
    assert decide([0.4, 0.4, 0.1, 0.1]) == "EMERGENCY"                   # tie goes to the more urgent class
    assert decide([0.3, 0.1, 0.5, 0.1], emergency_threshold=0.25) == "EMERGENCY"
    assert decide([0.3, 0.1, 0.5, 0.1], emergency_threshold=0.5) == "ROUTINE"


def test_normalise_collapses_whitespace_and_unicode():
    assert normalise("  chest   pain \n now ") == "chest pain now"
    assert normalise("café") == normalise("café")                 # composed == decomposed
    assert normalise(None) == ""


def test_cache_hit_avoids_backend_and_survives_restart(tmp_path):
    path = tmp_path / "gate.jsonl"
    b = FakeBackend({"chest pain": EMERG})
    g = CachedGate(b, "g1-s13", path)
    r1 = g.predict("chest pain")
    assert r1.label == "EMERGENCY" and r1.source == "live"
    r2 = g.predict("  chest   pain ")                                       # same text after normalisation
    assert r2.source == "cache" and len(b.calls) == 1
    g2 = CachedGate(FakeBackend(), "g1-s13", path)                         # new process, same cache file
    assert g2.predict("chest pain").source == "cache" and g2.predict("chest pain").label == "EMERGENCY"


def test_checkpoint_id_separates_caches(tmp_path):
    path = tmp_path / "gate.jsonl"
    CachedGate(FakeBackend({"x": EMERG}), "g1-s13", path).predict("x")
    other = CachedGate(FakeBackend({"x": [0.1, 0.1, 0.7, 0.1]}), "g1-s42", path)
    assert other.predict("x").label == "ROUTINE"


def test_batch_fills_only_missing_and_dedupes(tmp_path):
    b = FakeBackend()
    g = CachedGate(b, "c", tmp_path / "g.jsonl")
    g.predict("a")
    res = g.predict_many(["a", "b", "b", "c"])
    assert [r.source for r in res] == ["cache", "live", "live", "live"]
    assert b.calls[-1] == ["b", "c"]
    assert g.warm(["a", "b", "z"]) == 1


def test_fail_closed_and_failures_not_cached(tmp_path):
    path = tmp_path / "g.jsonl"
    g = CachedGate(BrokenBackend(), "c", path)
    r = g.predict("anything")
    assert r.label == UNAVAILABLE and r.outage and r.source == "fail_closed" and r.probs is None
    assert not path.exists() or path.read_text(encoding="utf-8") == ""
    assert CachedGate(None, "c", path).predict("anything").outage            # cache-only gate: miss fails closed


def test_invalid_probabilities_fail_closed(tmp_path):
    for bad in ([0.5, 0.5], [float("nan"), 0.5, 0.25, 0.25], [2.0, -0.5, -0.25, -0.25], [0.1, 0.1, 0.1, 0.1]):
        g = CachedGate(FakeBackend(default=bad), "c", tmp_path / f"{abs(hash(str(bad)))}.jsonl")
        assert g.predict("x").outage


def test_torn_cache_line_is_ignored(tmp_path):
    path = tmp_path / "g.jsonl"
    g = CachedGate(FakeBackend({"x": EMERG}), "c", path)
    g.predict("x")
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"key": "abc", "te')                                         # killed mid-write
    assert GateCache(path).get(g.key("x")) == EMERG


def test_checkpoint_id_changes_with_the_adapter_files(tmp_path):
    from swasthya.gate import checkpoint_id_from_dir, prompted_checkpoint_id
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"weights-one")
    a = checkpoint_id_from_dir(d, "g1")
    assert a.startswith("g1-") and checkpoint_id_from_dir(d, "g1") == a
    (d / "adapter_model.safetensors").write_bytes(b"weights-two")
    assert checkpoint_id_from_dir(d, "g1") != a
    (d / "adapter_model.safetensors").unlink()
    try:
        checkpoint_id_from_dir(d)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
    s = [{"id": "a", "text": "t", "label": "ROUTINE"}]
    assert prompted_checkpoint_id("m", s) == prompted_checkpoint_id("m", s)
    assert prompted_checkpoint_id("m", s) != prompted_checkpoint_id("m", []) != prompted_checkpoint_id("n", [])


def test_threshold_applies_to_cached_probs_without_recompute(tmp_path):
    b = FakeBackend({"x": [0.3, 0.1, 0.5, 0.1]})
    g = CachedGate(b, "c", tmp_path / "g.jsonl")
    assert g.predict("x").label == "ROUTINE"
    g.emergency_threshold = 0.25
    assert g.predict("x").label == "EMERGENCY" and len(b.calls) == 1
    assert json.loads((tmp_path / "g.jsonl").read_text(encoding="utf-8").splitlines()[0])["probs"] == b.table["x"]
