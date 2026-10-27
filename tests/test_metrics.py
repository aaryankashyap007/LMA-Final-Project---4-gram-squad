from swasthya.metrics import emergency_recall, over_triage_rate, macro_f1


def test_emergency_recall():
    y = ["EMERGENCY", "EMERGENCY", "ROUTINE"]
    p = ["EMERGENCY", "ROUTINE", "ROUTINE"]
    assert emergency_recall(y, p) == 0.5
    assert over_triage_rate(y, p) == 0.0


def test_macro_f1_perfect():
    y = ["EMERGENCY", "URGENT", "ROUTINE", "INSUFFICIENT"]
    assert macro_f1(y, y) == 1.0
