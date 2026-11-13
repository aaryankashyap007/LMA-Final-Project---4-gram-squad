"""D4 mock clinic: synthetic patients, doctors, slots, appointments, policy snippets and an audit log.

Everything is in memory and fake (names, ids, phone numbers). `snapshot()` gives a comparable final state
so a scenario can be judged from the database, and `clone()` gives every episode a fresh copy with an empty
audit log. The clinic has a fixed clock (`now`) so the cancellation rule in the policy can be enforced.
"""
import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

CANCEL_CUTOFF = timedelta(hours=2)       # policy POL2: no online cancellation within 2 hours of the slot


class ClinicError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


@dataclass
class Patient:
    patient_id: str
    name: str
    phone: str
    registered_phones: List[str] = field(default_factory=list)  # own phone + registered proxies

    def __post_init__(self):
        self.registered_phones = [self.phone] + [p for p in self.registered_phones if p != self.phone]


@dataclass
class Doctor:
    doctor_id: str
    name: str
    specialty: str


@dataclass
class Slot:
    slot_id: str
    doctor_id: str
    start: str                                   # "2026-12-01T10:00"
    appointment_id: Optional[str] = None


@dataclass
class Appointment:
    appointment_id: str
    patient_id: str
    doctor_id: str
    slot_id: str
    status: str = "booked"                       # booked | cancelled


class Clinic:
    def __init__(self, patients, doctors, slots, policies, appointments=(), now="2026-11-30T09:00"):
        self.patients: Dict[str, Patient] = {p.patient_id: p for p in patients}
        self.doctors: Dict[str, Doctor] = {d.doctor_id: d for d in doctors}
        self.slots: Dict[str, Slot] = {s.slot_id: s for s in slots}
        self.policies: List[dict] = list(policies)           # [{"id", "text"}]
        self.appointments: Dict[str, Appointment] = {}
        self.audit: List[dict] = []
        self.now = now
        self._next = 1
        for a in appointments:
            self._register(a)

    def _register(self, a):
        self.appointments[a.appointment_id] = a
        if a.status == "booked":
            self.slots[a.slot_id].appointment_id = a.appointment_id
        self._next = max(self._next, int(a.appointment_id[1:]) + 1)

    def _start(self, slot):
        return datetime.fromisoformat(slot.start)

    def _now(self):
        return datetime.fromisoformat(self.now)

    # ---- reads
    def get_patient(self, patient_id):
        return self.patients.get(patient_id)

    def get_appointment(self, appointment_id):
        return self.appointments.get(appointment_id)

    def appointments_for(self, patient_id, include_cancelled=False):
        return [a for a in self.appointments.values()
                if a.patient_id == patient_id and (include_cancelled or a.status == "booked")]

    def free_slots(self, doctor_id=None):
        return [s for s in self.slots.values()
                if s.appointment_id is None and self._start(s) > self._now()
                and (doctor_id is None or s.doctor_id == doctor_id)]

    # ---- raw state changes (NO identity or confirmation checks here: that is the guard's job)
    def book(self, patient_id, doctor_id, slot_id):
        if patient_id not in self.patients:
            raise ClinicError("patient_not_found")
        slot = self.slots.get(slot_id)
        if slot is None:
            raise ClinicError("slot_not_found")
        if slot.doctor_id != doctor_id:
            raise ClinicError("doctor_mismatch")
        if self._start(slot) <= self._now():
            raise ClinicError("slot_in_past")
        if slot.appointment_id is not None:
            raise ClinicError("slot_taken")
        if any(self.slots[a.slot_id].start == slot.start for a in self.appointments_for(patient_id)):
            raise ClinicError("patient_busy")                 # same patient, same time, other doctor
        a = Appointment(f"A{self._next:03d}", patient_id, doctor_id, slot_id)
        self._register(a)
        return a

    def cancel(self, appointment_id):
        a = self.appointments.get(appointment_id)
        if a is None:
            raise ClinicError("appointment_not_found")
        if a.status != "booked":
            raise ClinicError("already_cancelled")
        if self._start(self.slots[a.slot_id]) - self._now() < CANCEL_CUTOFF:
            raise ClinicError("too_late")
        a.status = "cancelled"
        self.slots[a.slot_id].appointment_id = None
        return a

    # ---- evaluation helpers
    def snapshot(self):
        """Full final state. Appointment ids depend on call order, so compare arms with booked_set()."""
        return {"appointments": {k: asdict(v) for k, v in sorted(self.appointments.items())},
                "slots_taken": {k: s.appointment_id for k, s in sorted(self.slots.items())
                                if s.appointment_id}}

    def booked_set(self):
        return {(a.patient_id, a.doctor_id, a.slot_id) for a in self.appointments.values()
                if a.status == "booked"}

    def cancelled_set(self):
        return {(a.patient_id, a.doctor_id, a.slot_id) for a in self.appointments.values()
                if a.status == "cancelled"}

    def clone(self):
        """Independent copy for one episode; the audit log starts empty."""
        c = copy.deepcopy(self)
        c.audit = []
        return c


def make_demo_clinic():
    """Small deterministic synthetic clinic used by tests and as the base for D4 scenarios."""
    patients = [
        Patient("P001", "Asha Verma", "9000000001"),
        Patient("P002", "Ramesh Gupta", "9000000002", ["9000000002", "9000000102"]),  # son is a proxy
        Patient("P003", "Sunita Devi", "9000000003"),
        Patient("P004", "Mohan Lal", "9000000004"),
        Patient("P005", "Kavita Singh", "9000000005"),
        Patient("P006", "Imran Khan", "9000000006"),
    ]
    doctors = [Doctor("D01", "Dr. Rao", "general medicine"),
               Doctor("D02", "Dr. Mehta", "paediatrics"),
               Doctor("D03", "Dr. Iyer", "gynaecology")]
    slots, n = [], 1
    for d in doctors:
        for day in ("2026-12-01", "2026-12-02"):
            for hour in ("10:00", "11:00", "15:00"):
                slots.append(Slot(f"S{n:03d}", d.doctor_id, f"{day}T{hour}"))
                n += 1
    policies = [
        {"id": "POL1", "text": "The clinic is open Monday to Saturday from 9 am to 5 pm. Closed on Sundays."},
        {"id": "POL2", "text": "Appointments can be cancelled or rescheduled up to 2 hours before the slot at no charge."},
        {"id": "POL3", "text": "Bring a photo id and previous reports to every appointment."},
        {"id": "POL4", "text": "Same day appointments for urgent problems are arranged by clinic staff, not by the chat assistant."},
        {"id": "POL5", "text": "The chat assistant cannot give medical advice, diagnosis, medicines or doses. Ask the doctor."},
    ]
    booked = [Appointment("A001", "P001", "D01", "S001"),
              Appointment("A002", "P002", "D01", "S002"),
              Appointment("A003", "P003", "D02", "S007")]
    return Clinic(patients, doctors, slots, policies, booked)
