import pytest

from swasthya.clinic import ClinicError, make_demo_clinic
from swasthya.tools import run_tool, validate_args


def test_book_and_cancel_change_state():
    c = make_demo_clinic()
    before = c.snapshot()
    a = c.book("P004", "D01", "S003")
    assert ("P004", "D01", "S003") in c.booked_set()
    assert c.slots["S003"].appointment_id == a.appointment_id
    c.cancel(a.appointment_id)
    assert ("P004", "D01", "S003") not in c.booked_set()
    assert c.slots["S003"].appointment_id is None
    assert c.snapshot() != before            # the cancelled appointment stays on record


def test_booking_errors():
    c = make_demo_clinic()
    with pytest.raises(ClinicError) as e:
        c.book("P004", "D01", "S001")        # S001 is taken by A001
    assert e.value.code == "slot_taken"
    with pytest.raises(ClinicError) as e:
        c.book("P004", "D02", "S003")        # S003 belongs to D01
    assert e.value.code == "doctor_mismatch"
    with pytest.raises(ClinicError) as e:
        c.cancel("A999")
    assert e.value.code == "appointment_not_found"


def test_clone_is_independent():
    c = make_demo_clinic()
    d = c.clone()
    d.book("P004", "D01", "S003")
    assert ("P004", "D01", "S003") not in c.booked_set()


def test_schema_validation():
    assert validate_args("book_appointment", {"patient_id": "P001", "doctor_id": "D01", "slot_id": "S003"}) == []
    assert validate_args("book_appointment", {"patient_id": "P001"})                      # missing args
    assert validate_args("book_appointment", {"patient_id": "bob", "doctor_id": "D01", "slot_id": "S003"})
    assert validate_args("cancel_appointment", {"appointment_id": "A001", "extra": "x"})
    assert validate_args("lookup_policy", {"query": ""})
    assert validate_args("drop_database", {})


def test_tools_return_generic_errors():
    c = make_demo_clinic()
    r = run_tool(c, "book_appointment", {"patient_id": "P004", "doctor_id": "D01", "slot_id": "S001"})
    assert not r.ok and r.error == "slot_taken"
    assert "P001" not in r.message and "Asha" not in r.message     # never reveal who holds the slot
    r = run_tool(c, "cancel_appointment", {"appointment_id": "A999"})
    assert not r.ok


def test_lookup_appointments_and_policy():
    c = make_demo_clinic()
    r = run_tool(c, "lookup_appointments", {"patient_id": "P001"})
    assert r.ok and [a["appointment_id"] for a in r.data["appointments"]] == ["A001"]
    r = run_tool(c, "lookup_policy", {"query": "can I cancel my appointment"})
    assert r.ok and not r.data["abstain"] and r.data["snippets"][0]["id"] == "POL2"
    r = run_tool(c, "lookup_policy", {"query": "zzzz qqqq"})
    assert r.ok and r.data["abstain"] and r.data["snippets"] == []
