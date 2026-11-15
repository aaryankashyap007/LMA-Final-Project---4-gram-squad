import csv
import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_review_sheets", ROOT / "scripts/make_review_sheets.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def read(path):
    return list(csv.reader(open(path, encoding="utf-8-sig")))


def test_committed_sheets_are_in_sync_with_the_generator():
    h, rows = mod.native_sheet(random.Random(mod.SEED))
    assert read(mod.OUT / "review_native.csv") == [h] + rows
    h2, rows2, key = mod.clinician_sheet(random.Random(mod.SEED))
    assert read(mod.OUT / "review_clinician.csv") == [h2] + rows2
    assert read(mod.OUT / "_answer_key_clinician.csv") == [["case_id", "our_label", "pair_id"]] + key


def test_the_clinician_sheet_is_blind():
    h, rows, key = mod.clinician_sheet(random.Random(mod.SEED))
    labelled = [r for r in rows if r[0] == "label this message"]
    assert len(labelled) == 40 and len(key) == 40
    assert all(r[3] == "" and r[5] == "" for r in labelled)            # no category hint, no answer
    assert {k[0] for k in key} == {r[1] for r in labelled}


def test_native_sheet_has_every_script_for_every_scenario_row():
    h, rows = mod.native_sheet(random.Random(mod.SEED))
    for r in rows:
        if r[0] in ("scenario", "planted note"):
            assert r[3] and r[4] and r[5], r[:2]


def test_scoring_reports_agreement_and_kappa(tmp_path, capsys):
    h, rows, key = mod.clinician_sheet(random.Random(mod.SEED))
    ours = {k[0]: k[1] for k in key}
    flipped = [k[0] for k in key if k[1] == "URGENT"][:2]
    filled = tmp_path / "filled.csv"
    with open(filled, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(h)
        for r in rows:
            if r[0] == "label this message":
                r = r[:5] + [("ROUTINE" if r[1] in flipped else ours[r[1]]), ""]
            w.writerow(r)
    mod.score(str(filled))
    out = capsys.readouterr().out
    assert "40 labelled cases; exact agreement 38/40" in out and "Cohen's kappa" in out
    assert all(c in out for c in flipped)
