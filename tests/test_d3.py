from swasthya import d3


def row(cid, label="ROUTINE", pair="", **kw):
    r = dict(case_id=cid, category="general_medical", label=label, age="30", sex="M",
             cues="cough;one_week", en="I have had a cough for one week.",
             deva="मुझे एक हफ्ते से खांसी है।", hinglish="mujhe ek hafte se khansi hai",
             pair_id=pair, notes="")
    r.update(kw)
    return r


def test_valid_rows_pass():
    errs, _ = d3.validate_rows([row("A1"), row("A2")])
    assert errs == []


def test_bad_rows_flagged():
    errs, _ = d3.validate_rows([row("A1", deva="mujhe khansi hai bahut"), row("A1", label="BAD")])
    joined = " ".join(errs)
    assert "Devanagari" in joined and "duplicate" in joined and "bad label" in joined


def test_pair_must_differ_and_stay_together():
    rows = [row("P1", "EMERGENCY", "X-P1"), row("P2", "ROUTINE", "X-P1")] + \
           [row(f"C{i}", "EMERGENCY") for i in range(10)]
    assert d3.validate_rows(rows)[0] == []
    d3.assign_splits(rows)
    assert rows[0]["split"] == rows[1]["split"]
    bad = [row("P1", "ROUTINE", "Z"), row("P2", "ROUTINE", "Z")]
    assert any("must differ" in e for e in d3.validate_rows(bad)[0])


def test_parity_report():
    r = row("A1", a2_present_en="cough;one_week", a2_present_deva="cough", a2_present_hinglish="cough;one_week", a2_extra="no")
    rep = d3.parity_report([r])
    assert rep["en"][0] == 1.0 and rep["deva"][0] == 0.0
