from swasthya import kio, prompts
from swasthya.schema import LABELS


def test_letters_match_labels():
    assert len(prompts.LETTERS) == len(LABELS) == 4


def test_messages_zero_and_few_shot():
    m = prompts.build_messages("hello")
    assert [x["role"] for x in m] == ["system", "user"] and "hello" in m[-1]["content"]
    m = prompts.build_messages("q", shots=[{"text": "x", "label": "ROUTINE"}])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]
    assert m[2]["content"] == "C"


def test_pick_shots_balanced_and_deterministic():
    rows = [{"id": f"{l}{i}", "text": "t", "label": l} for l in LABELS for i in range(10)]
    a, b = prompts.pick_shots(rows, 8, 13), prompts.pick_shots(rows, 8, 13)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert sorted(r["label"] for r in a) == sorted(LABELS * 2)
    assert prompts.pick_shots(rows, 0) == []


def test_pred_rows_argmax():
    rows = [{"id": "a"}, {"id": "b"}]
    out = kio.pred_rows(rows, [[0.1, 0.7, 0.1, 0.1], [0.9, 0.05, 0.03, 0.02]])
    assert [o["pred"] for o in out] == ["URGENT", "EMERGENCY"]
