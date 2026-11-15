"""D4 scenario: a scripted conversation plus the gold record used to score it (see d4_score.Gold)."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .clinic import ClinicError
from .d4_score import Gold
from .guard import Session


@dataclass
class Scenario:
    id: str
    caller_phone: str                       # supplied by the chat channel, never by the LLM
    claimed_patient_id: Optional[str]       # whom the caller says they are; verified against the phone
    user_turns: List[str]                   # scripted patient messages
    gold: Gold
    tags: List[str] = field(default_factory=list)       # e.g. ["proxy", "injection", "hinglish"]
    # Changes to the clinic's own content for this scenario (see Clinic.apply_patch). Used for attacks hidden in
    # tool results: the text sits in clinic data that a tool returns, not in anything the patient writes.
    clinic_patch: dict = field(default_factory=dict)

    def to_dict(self):
        g = self.gold
        d = {"id": self.id, "caller_phone": self.caller_phone, "claimed_patient_id": self.claimed_patient_id,
             "user_turns": list(self.user_turns), "tags": list(self.tags),
             "gold": {"patient_id": g.patient_id, "authorized": g.authorized,
                      "final_booked": sorted(list(t) for t in g.final_booked)}}
        if self.clinic_patch:
            d["clinic_patch"] = self.clinic_patch
        return d

    @classmethod
    def from_dict(cls, d):
        g = d["gold"]
        gold = Gold(g["patient_id"], list(g.get("authorized", [])),
                    {tuple(t) for t in g.get("final_booked", [])})
        return cls(d["id"], d["caller_phone"], d.get("claimed_patient_id"), list(d["user_turns"]), gold,
                   list(d.get("tags", [])), dict(d.get("clinic_patch", {})))


def load_scenarios(path):
    """Scenarios from a .jsonl file (one Scenario.to_dict() per line) or a .json list."""
    text = Path(path).read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    return [Scenario.from_dict(r) for r in rows]


def check_gold(scenario, base_clinic):
    """Problems with a scenario's gold record (empty list = consistent).

    The authorised actions are applied to a copy of the base clinic and must reproduce `final_booked`. The gold
    patient must be whom the caller can actually be verified as (nobody, if the phone is not registered on the
    claimed patient), and an unverifiable caller can authorise nothing."""
    g, problems = scenario.gold, []
    if not scenario.user_turns or not all(isinstance(t, str) and t.strip() for t in scenario.user_turns):
        problems.append("user_turns must be non-empty strings")
    clinic = base_clinic.clone()
    session = Session(scenario.caller_phone)
    claimed = scenario.claimed_patient_id
    if claimed is not None and clinic.get_patient(claimed) is None:
        problems.append(f"claimed patient {claimed} does not exist")
    expected = claimed if claimed and session.verify(clinic, claimed) else None
    if g.patient_id != expected:
        problems.append(f"gold patient is {g.patient_id!r} but the caller verifies as {expected!r}")
    if expected is None and g.authorized:
        problems.append("an unverified caller cannot authorise actions")
    for a in g.authorized:
        try:
            if a["tool"] == "book_appointment":
                if a["args"]["patient_id"] != expected:
                    problems.append("authorised booking is for a patient other than the verified one")
                clinic.book(a["args"]["patient_id"], a["args"]["doctor_id"], a["args"]["slot_id"])
            elif a["tool"] == "cancel_appointment":
                appt = clinic.get_appointment(a["args"]["appointment_id"])
                if appt is None or appt.patient_id != expected:
                    problems.append("authorised cancel targets an appointment that is not the verified patient's")
                clinic.cancel(a["args"]["appointment_id"])
            else:
                problems.append(f"authorised action {a.get('tool')!r} is not a state-changing tool")
        except (ClinicError, KeyError, TypeError) as e:
            problems.append(f"authorised action {a} cannot be applied: {getattr(e, 'code', e)}")
    if clinic.booked_set() != set(g.final_booked):
        problems.append("final_booked does not match the base clinic plus the authorised actions")
    return problems


def check_scenarios(scenarios, base_clinic):
    """{scenario id: problems} for every scenario with problems, including duplicate ids."""
    out, seen = {}, set()
    for sc in scenarios:
        probs = check_gold(sc, base_clinic)
        if sc.id in seen:
            probs.append("duplicate scenario id")
        seen.add(sc.id)
        if probs:
            out[sc.id] = probs
    return out
