import itertools

import pytest

from swasthya import guard
from swasthya.clinic import make_demo_clinic
from swasthya.mutation import MUTANTS, apply, make_evaluate
from swasthya.guard import Session, action_key

REAL_EVALUATE = guard.evaluate          # captured at import, before any mutant is applied by a test


def _situations():
    """A grid of (session, tool, args) covering every guard branch."""
    clinic = make_demo_clinic()
    calls = [("lookup_policy", {"query": "opening hours"}),
             ("lookup_appointments", {"patient_id": "P004"}), ("lookup_appointments", {"patient_id": "P001"}),
             ("lookup_appointments", {}),
             ("cancel_appointment", {"appointment_id": "A001"}), ("cancel_appointment", {"appointment_id": "A999"}),
             ("book_appointment", {"patient_id": "P004", "doctor_id": "D01", "slot_id": "S003"}),
             ("book_appointment", {"patient_id": "P001", "doctor_id": "D01", "slot_id": "S003"}),
             ("nonexistent_tool", {}), ("escalate_to_staff", {"urgency": "URGENT"})]
    for pid, locked, confirmed_key in itertools.product(
            (None, "P004", "P001"), (False, True), (None, "book", "other")):
        for tool, args in calls:
            s = Session("9000000004")
            s.patient_id = pid
            s.booking_locked = locked
            if confirmed_key == "book":
                s.confirmed = action_key(tool, args)
            elif confirmed_key == "other":
                s.confirmed = action_key("cancel_appointment", {"appointment_id": "A002"})
            yield clinic, s, tool, args


def test_restated_evaluate_matches_the_real_guard_when_nothing_is_skipped():
    mine = make_evaluate()
    n = 0
    for clinic, s, tool, args in _situations():
        real, got = REAL_EVALUATE(clinic, s, tool, args), mine(clinic, s, tool, args)
        assert (real.allowed, real.reason) == (got.allowed, got.reason), (tool, args, s)
        n += 1
    assert n == 180


def test_every_mutant_changes_the_guard_and_apply_undoes_it():
    before = (guard.evaluate, guard.is_affirmation, guard.Session.verify)
    for name in MUTANTS:
        undo = apply(name)
        undo()
        assert (guard.evaluate, guard.is_affirmation, guard.Session.verify) == before, name


@pytest.mark.parametrize("name", [n for n in MUTANTS if n not in ("loose_affirmation", "verify_any_phone")])
def test_each_evaluate_mutant_gives_a_different_verdict_somewhere(name):
    """A mutant that behaves exactly like the real guard on every branch would be an equivalent mutant."""
    patch = MUTANTS[name][1][0][2]
    differs = any((REAL_EVALUATE(c, s, t, a).allowed, REAL_EVALUATE(c, s, t, a).reason)
                  != (patch(c, s, t, a).allowed, patch(c, s, t, a).reason) for c, s, t, a in _situations())
    assert differs, name


from swasthya.mutation import PROBES, run_probe, run_probes  # noqa: E402


def test_no_probe_harms_with_the_real_guard():
    assert not any(run_probes().values())


@pytest.mark.parametrize("mutant,probe", [
    ("no_ownership", "yes_to_cancel_of_another_patients_appointment"),
    ("no_patient_mismatch", "yes_to_booking_for_another_patient"),
    ("no_patient_mismatch", "yes_to_a_lookup_of_another_patient_is_not_needed"),
    ("verify_any_phone", "yes_to_cancel_by_an_unverified_caller"),
    ("loose_affirmation", "non_yes_message_after_a_proposal"),
    ("confirm_any_action", "yes_to_one_action_then_a_different_one"),
    ("no_confirmation", "non_yes_message_after_a_proposal"),
])
def test_probe_catches_the_mutant_it_is_meant_for(mutant, probe):
    undo = apply(mutant)
    try:
        assert run_probe(probe)
    finally:
        undo()
    assert not run_probe(probe)


def test_pair_mutants_exist_and_are_named_after_their_parts():
    pairs = [n for n in MUTANTS if "+" in n]
    assert len(pairs) == 6 and all(n.startswith("no_confirmation+") for n in pairs)
    assert set(PROBES) >= {"yes_to_cancel_by_an_unverified_caller"}
