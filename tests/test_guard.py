from swasthya.clinic import make_demo_clinic
from swasthya.guard import Executor, Session, evaluate, is_affirmation

BOOK = {"patient_id": "P004", "doctor_id": "D01", "slot_id": "S003"}


def verified(phone="9000000004", pid="P004"):
    c = make_demo_clinic()
    s = Session(phone)
    assert s.verify(c, pid)
    return c, s


def test_affirmation_en_hi_hinglish():
    for t in ["yes", "Yes please", "haan ji", "हाँ", "ठीक है", "ji kar dijiye", "ok", "sure go ahead",
              "haan bilkul"]:
        assert is_affirmation(t), t
    for t in ["no", "nahi", "नहीं", "yes but change the time", "haan lekin dusra slot", "", "what time is it",
              "yes " * 10, "mujhe cancel kar do", "please cancel", "ok book Dr Mehta tomorrow", "kar do",
              "cancel everything please"]:
        assert not is_affirmation(t), t


def test_verify_own_phone_and_proxy_only():
    c = make_demo_clinic()
    assert Session("9000000002").verify(c, "P002")          # patient's own phone
    assert Session("9000000102").verify(c, "P002")          # registered proxy
    assert not Session("9000000999").verify(c, "P002")      # stranger
    assert not Session("9000000102").verify(c, "P001")      # proxy of someone else
    assert not Session("9000000002").verify(c, "P404")      # no such patient


def test_unverified_gets_policy_only():
    c = make_demo_clinic()
    s = Session("9000000999")
    assert evaluate(c, s, "lookup_policy", {"query": "opening hours"}).allowed
    assert evaluate(c, s, "lookup_appointments", {"patient_id": "P001"}).reason == "not_verified"
    assert evaluate(c, s, "book_appointment", BOOK).reason == "not_verified"


def test_read_scoped_to_verified_patient():
    c, s = verified()
    assert evaluate(c, s, "lookup_appointments", {"patient_id": "P004"}).allowed
    assert evaluate(c, s, "lookup_appointments", {"patient_id": "P001"}).reason == "patient_mismatch"


def test_someone_elses_appointment_looks_exactly_like_a_missing_one():
    c, s = verified()
    assert evaluate(c, s, "cancel_appointment", {"appointment_id": "A001"}).reason == "not_owner"
    ex = Executor(c, s, guard_on=True)
    theirs = ex.call("cancel_appointment", {"appointment_id": "A001"})
    missing = ex.call("cancel_appointment", {"appointment_id": "A999"})
    assert not theirs.ok and (theirs.error, theirs.message) == (missing.error, missing.message)
    assert "Asha" not in theirs.message and "P001" not in theirs.message
    assert c.get_appointment("A001").status == "booked"


def test_confirmation_flow_allows_exactly_one_action():
    c, s = verified()
    ex = Executor(c, s, guard_on=True)
    r = ex.call("book_appointment", BOOK)
    assert not r.ok and r.error == "needs_confirmation" and ("P004", "D01", "S003") not in c.booked_set()
    s.on_user_message("haan ji")
    r = ex.call("book_appointment", BOOK)
    assert r.ok and ("P004", "D01", "S003") in c.booked_set()
    r = ex.call("book_appointment", {**BOOK, "slot_id": "S004"})       # confirmation was consumed
    assert r.error == "needs_confirmation"


def test_confirmation_message_carries_a_code_generated_summary():
    c, s = verified()
    r = Executor(c, s).call("book_appointment", BOOK)
    summary = r.data["summary"]
    assert "Dr. Rao" in summary and "2026-12-01 at 15:00" in summary and "Mohan Lal" in summary
    assert summary in r.message and s.pending_summary == summary
    s.on_user_message("nahi")
    assert s.pending_summary is None


def test_confirmation_does_not_carry_to_different_args():
    c, s = verified()
    ex = Executor(c, s, guard_on=True)
    ex.call("book_appointment", BOOK)
    s.on_user_message("yes")
    r = ex.call("book_appointment", {**BOOK, "slot_id": "S004"})       # a different slot than proposed
    assert r.error == "needs_confirmation" and len(c.booked_set()) == 3


def test_non_affirmation_clears_pending():
    c, s = verified()
    ex = Executor(c, s, guard_on=True)
    ex.call("book_appointment", BOOK)
    s.on_user_message("nahi")
    assert ex.call("book_appointment", BOOK).error == "needs_confirmation"
    assert ("P004", "D01", "S003") not in c.booked_set()


def test_new_request_after_proposal_is_not_a_confirmation():
    c, s = verified()
    ex = Executor(c, s, guard_on=True)
    ex.call("book_appointment", BOOK)
    s.on_user_message("ok book Dr Mehta tomorrow instead")
    assert ex.call("book_appointment", BOOK).error == "needs_confirmation"


def test_lock_blocks_state_changes_but_not_reads():
    c, s = verified()
    ex = Executor(c, s, guard_on=True)
    ex.call("book_appointment", BOOK)
    s.on_user_message("yes")
    s.lock_booking()
    assert s.pending is None and s.pending_summary is None and s.confirmed is None
    assert ex.call("book_appointment", BOOK).error == "tool_locked"
    assert ex.call("lookup_appointments", {"patient_id": "P004"}).ok
    assert ("P004", "D01", "S003") not in c.booked_set()


def test_proxy_can_act_for_the_registered_patient():
    c = make_demo_clinic()
    s = Session("9000000102")
    assert s.verify(c, "P002")
    ex = Executor(c, s)
    args = {"patient_id": "P002", "doctor_id": "D01", "slot_id": "S004"}
    assert ex.call("book_appointment", args).error == "needs_confirmation"
    s.on_user_message("haan")
    assert ex.call("book_appointment", args).ok
    assert ex.call("lookup_appointments", {"patient_id": "P001"}).error == "patient_mismatch"


def test_guard_off_executes_and_records_diagnostics():
    c, s = verified()
    ex = Executor(c, s, guard_on=False)
    assert ex.call("cancel_appointment", {"appointment_id": "A001"}).ok      # someone else's appointment
    assert c.get_appointment("A001").status == "cancelled"
    assert ex.log[-1]["unapproved"] and ex.log[-1]["verdict"] == "not_owner"
    r = ex.call("lookup_appointments", {"patient_id": "P002"})               # another patient's records
    assert r.ok and ex.log[-1]["leak"]
    assert ex.call("book_appointment", BOOK).ok                              # skipped confirmation
    assert ex.log[-1]["unapproved"] and ex.log[-1]["verdict"] == "needs_confirmation"


def test_malformed_tool_names_and_args_do_not_crash():
    c, s = verified()
    assert evaluate(c, s, ["x"], {}).reason == "unknown_tool"
    assert evaluate(c, s, "book_appointment", "not a dict").reason == "schema_invalid"
    ex = Executor(c, s, guard_on=False)
    ex.call(["x"], {})
    ex.call("book_appointment", None)
    assert all(isinstance(e["tool"], str) for e in ex.log)
    assert c.booked_set() == make_demo_clinic().booked_set()


def test_session_turn_and_single_audit_log():
    c, s = verified()
    ex = Executor(c, s, guard_on=True)
    s.on_user_message("book me a slot")
    ex.call("lookup_policy", {"query": "opening hours"})
    ex.call("book_appointment", BOOK)
    assert ex.log is c.audit and len(c.audit) == 2
    assert c.audit[0]["turn"] == 1 and c.audit[1]["verdict"] == "needs_confirmation"
