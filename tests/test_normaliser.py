import json
from pathlib import Path

import pytest

from swasthya.normaliser import detect_script, for_match, for_model

D3 = Path(__file__).resolve().parents[1] / "data/d3_triage/rendered"


def test_for_model_removes_invisible_characters_and_fixes_digits_and_quotes():
    assert for_model("be​hosh‍ ﻿hai") == "behosh hai"             # zero-width characters can hide text
    assert for_model("बुखार १०२ डिग्री") == "बुखार 102 डिग्री"                      # Devanagari digits
    assert for_model("temp ١٠٢") == "temp 102"                    # Arabic-Indic digits
    assert for_model("ＣＨＥＳＴ ＡＰＮ") == "CHEST APN"   # full-width letters (NFKC)
    assert for_model("it’s “bad” – now") == "it's \"bad\" - now"
    assert for_model("a\x00b\x07c") == "abc"                                      # control characters


def test_for_model_keeps_line_breaks_danda_and_is_idempotent():
    t = "Patient: मुझे दर्द है।\nAssistant: कब से?"
    assert for_model(t) == t
    assert for_model("  a   b \n\n  c ") == "a b\nc"
    assert for_model(for_model("x​  y\r\nz")) == for_model("x​  y\r\nz")
    assert for_model(None) == "" and for_model("") == ""


def test_nukta_forms_are_unified():
    precomposed, decomposed = "ज़", "ज़"                          # ज़ both ways
    assert for_model(precomposed) == for_model(decomposed)
    assert for_match("ज़हर") == for_match("जहर")


def test_for_match_is_spelling_insensitive_for_chat_text():
    assert for_match("Saaaans NHI aa rhi!!") == "saans nahi aa rahi"
    assert for_match("can't breathe") == "cant breathe"
    assert for_match("सीने में दर्द है।") == "सीने में दर्द है"
    assert for_match("हाँ") == for_match("हां")                                 # chandrabindu -> anusvara
    assert for_match("nhi", canon=False) == "nhi" and for_match("nhi") == "nahi"
    assert for_match("he is fine") == "he is fine"                               # English words are not respelled


@pytest.mark.parametrize("text,want", [
    ("ok", "unknown"), ("11", "unknown"), ("thanks", "unknown"), ("", "unknown"), ("???", "unknown"),
    ("11 am", "en"),                                                           # "am" is a real English word
    ("My son has a high fever since yesterday", "en"), ("Chest pain issue.", "en"),
    ("meri beti ko bahut tez bukhar hai", "hinglish"), ("haan ji kar dijiye", "hinglish"),
    ("mujhe bukhar hai aur I need a doctor", "hinglish"),
    ("मुझे तेज़ बुखार है", "deva"), ("मेरी बेटी बेहोश हो गई है", "deva"),
    ("मुझे fever है", "mixed")])
def test_detect_script(text, want):
    assert detect_script(text) == want


def d3_rows():
    out = []
    for sp in ("train", "dev", "test"):
        f = D3 / f"d3_{sp}.jsonl"
        if not f.exists():
            pytest.skip("D3 jsonl not present")
        out += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return out


def test_for_model_does_not_change_any_d3_text():
    """G1 was trained on raw D3 text, so the model-facing normalisation must not shift its input."""
    changed = [r["id"] for r in d3_rows() if for_model(r["text"]) != r["text"]]
    assert changed == []


def test_script_detection_never_confuses_languages_on_d3():
    wrong = [(r["script"], detect_script(r["text"])) for r in d3_rows() if detect_script(r["text"]) != r["script"]]
    assert all(got == "unknown" for _, got in wrong) and len(wrong) <= 6      # only very short English may be unknown
