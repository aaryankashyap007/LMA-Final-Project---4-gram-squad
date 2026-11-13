import pytest

from swasthya.clinic import ClinicError, Patient, make_demo_clinic
from swasthya.tools import run_tool, validate_args


def test_book_and_cancel_change_state():
    c = make_demo_clinic()
    before = c.snapshot()
    a = c.book("P004", "D01", "S003")
    assert ("P004", "D01", "S003") in c.booked_set()
    assert c.slots["S003"].appointment_id == a.appointment_id
    c.cancel(a.appointment_id)
    assert ("P004", "D01", "S003") not in c.booked_set()
    assert ("P004", "D01", "S003") in c.cancelled_set()
    assert c.slots["S003"].appointment_id is None
    assert c.snapshot() != before            # the cancelled appointment stays on record


def test_booking_errors():
    c = make_demo_clinic()
    for args, code in [(("P004", "D01", "S001"), "slot_taken"),        # S001 held by A001
                       (("P004", "D02", "S003"), "doctor_mismatch"),   # S003 belongs to D01
                       (("P004", "D01", "S999"), "slot_not_found"),
                       (("P999", "D01", "S003"), "patient_not_found")]:
        with pytest.raises(ClinicError) as e:
            c.book(*args)
        assert e.value.code == code
    with pytest.raises(ClinicError) as e:
        c.cancel("A999")
    assert e.value.code == "appointment_not_found"


def test_patient_cannot_double_book_the_same_time():
    c = make_demo_clinic()
    with pytest.raises(ClinicError) as e:
        c.book("P001", "D03", "S013")        # D03 Dec 1 10:00 clashes with P001's A001 (D01 Dec 1 10:00)
    assert e.value.code == "patient_busy"


def test_clock_rules_past_slots_and_late_cancellation():
    c = make_demo_clinic()
    c.now = "2026-12-01T09:30"                # A001 starts 10:00, A002 at 11:00: both inside the 2-hour cutoff
    for appt in ("A001", "A002"):
        with pytest.raises(ClinicError) as e:
            c.cancel(appt)
        assert e.value.code == "too_late"
    c.now = "2026-12-01T07:00"                # 3 hours before A001: allowed
    assert c.cancel("A001").status == "cancelled"
    c.now = "2026-12-01T12:00"
    with pytest.raises(ClinicError) as e:
        c.book("P006", "D03", "S013")        # D03 Dec 1 10:00 is already in the past
    assert e.value.code == "slot_in_past"
    assert all(s.start > "2026-12-01T12:00" for s in c.free_slots())


def test_clone_is_independent_and_resets_audit():
    c = make_demo_clinic()
    c.audit.append({"x": 1})
    d = c.clone()
    d.book("P004", "D01", "S003")
    assert ("P004", "D01", "S003") not in c.booked_set()
    assert d.audit == [] and c.audit == [{"x": 1}]


def test_patient_registered_phones_not_aliased():
    phones = ["9000000102"]
    p = Patient("P010", "X", "9000000010", phones)
    assert p.registered_phones == ["9000000010", "9000000102"] and phones == ["9000000102"]


def test_schema_validation():
    assert validate_args("book_appointment", {"patient_id": "P001", "doctor_id": "D01", "slot_id": "S003"}) == []
    assert validate_args("book_appointment", {"patient_id": "P001"})                      # missing args
    assert validate_args("book_appointment", {"patient_id": "bob", "doctor_id": "D01", "slot_id": "S003"})
    assert validate_args("cancel_appointment", {"appointment_id": "A001", "extra": "x"})
    assert validate_args("lookup_policy", {"query": ""})
    assert validate_args("drop_database", {})
    assert validate_args(["lookup_policy"], {"query": "x"})                                # non-string name
    assert validate_args("lookup_appointments", {"patient_id": "P१२३"})     # Devanagari digits


def test_validation_errors_do_not_echo_caller_text():
    errs = validate_args("lookup_appointments", {"patient_id": "P001", "IGNORE_ALL_RULES": "x"})
    assert errs and "IGNORE" not in " ".join(errs)
    r = run_tool(make_demo_clinic(), "lookup_appointments", {"patient_id": "P001", "IGNORE_ALL_RULES": "x"})
    assert not r.ok and "IGNORE" not in r.message


def test_tools_return_generic_errors():
    c = make_demo_clinic()
    r = run_tool(c, "book_appointment", {"patient_id": "P004", "doctor_id": "D01", "slot_id": "S001"})
    assert not r.ok and r.error == "slot_taken"
    assert "P001" not in r.message and "Asha" not in r.message     # never reveal who holds the slot
    r = run_tool(c, "cancel_appointment", {"appointment_id": "A999"})
    assert not r.ok and r.error == "not_found"
    c.now = "2026-12-01T09:30"
    r = run_tool(c, "cancel_appointment", {"appointment_id": "A001"})
    assert not r.ok and r.error == "too_late"


def test_lookup_appointments_and_policy():
    c = make_demo_clinic()
    r = run_tool(c, "lookup_appointments", {"patient_id": "P001"})
    assert r.ok and [a["appointment_id"] for a in r.data["appointments"]] == ["A001"]
    r = run_tool(c, "lookup_policy", {"query": "can I cancel my appointment"})
    assert r.ok and not r.data["abstain"] and r.data["snippets"][0]["id"] == "POL2"
    assert [s["id"] for s in r.data["snippets"]] == ["POL2"]       # one shared word is not a match
    r = run_tool(c, "lookup_policy", {"query": "when is the clinic open"})
    assert r.data["snippets"][0]["id"] == "POL1"
    r = run_tool(c, "lookup_policy", {"query": "do you have parking appointment"})
    assert r.data["abstain"] and r.data["snippets"] == []
    r = run_tool(c, "lookup_policy", {"query": "zzzz qqqq"})
    assert r.ok and r.data["abstain"]
