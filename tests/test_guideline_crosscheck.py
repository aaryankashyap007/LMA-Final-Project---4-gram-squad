import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("guideline_crosscheck", ROOT / "scripts/guideline_crosscheck.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_negation_guard_and_age_rule():
    assert "R07" in mod.fired("I have had chest pain for 40 minutes.", 66)
    assert "R07" not in mod.fired("I have had chest pain for 40 minutes.", 45)        # the IITT rule is for over 50
    assert "R07" not in mod.fired("I have no chest pain.", 66)
    assert mod.derive(mod.fired("He is not breathing and not responding.", 27)) == "red"


def test_paediatric_sheet_only_applies_under_twelve():
    text = "My 4 day old baby is very sleepy and is not feeding."
    assert "R13" in mod.fired(text, 0) and "R13" not in mod.fired(text, 30)


def test_every_disagreement_has_a_reading_and_the_headline_numbers_hold():
    rows = mod.load()
    for r in rows:
        r["ids"] = mod.fired(r["en"], r["age_n"])
        r["iitt"] = mod.derive(r["ids"])
    scored = [r for r in rows if r["label"] in mod.EXPECT]
    dis = [r for r in scored if mod.EXPECT[r["label"]] != r["iitt"]]
    assert len(scored) == 126 and len(dis) == 37
    assert all(r["case_id"] in mod.NOTES for r in dis)
    assert sum(mod.EXPECT[r["label"]] == r["iitt"] for r in scored) == 89
    assert sum((r["label"] == "EMERGENCY") == (r["iitt"] == "red") for r in scored) == 107
