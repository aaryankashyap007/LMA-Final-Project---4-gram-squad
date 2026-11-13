"""Deterministic action guard and the executor that runs tool calls through it.

The guard is plain code (no model). Before a call runs it checks, in this order:
  1. allow-list + schema     known tool, exact arguments, valid id formats
  2. identity                session is verified; the patient in the call is the verified patient; a cancel
                             targets one of the verified patient's own appointments (this also scopes reads)
  3. lock                    state-changing tools are disabled once the session escalated to staff
  4. confirmation            the exact same state-changing call was proposed and the patient then said yes

`evaluate` is pure (no side effects), so the same verdict can be recorded when the guard is switched off.
The audit flags `unapproved` / `leak` are DIAGNOSTICS of the guard's own rules (zero by construction when the
guard is on). The reported metrics must come from the independent oracle in d4_score.py, which judges what
ran against what the scenario says the patient authorised.

Confirmation protocol: a state-changing call without confirmation is refused with `needs_confirmation`, becomes
the session's pending action and gets a code-generated summary (`Session.pending_summary`). The harness must
show that summary to the patient (the LLM could otherwise describe a different action). If the patient's next
message is a short, unqualified yes (Session.on_user_message), the same call with identical arguments is
allowed once.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .clinic import Clinic
from .tools import GENERIC, TOOLS, ToolResult, run_tool, validate_args

IDENTITY_REASONS = {"not_verified", "patient_mismatch", "not_owner"}

# What the orchestrator is told on a block. Generic on purpose: never reveals other patients' data, and
# "someone else's appointment" reads exactly like "no such appointment".
USER_MESSAGES = {
    "unknown_tool": "That action is not available.",
    "schema_invalid": "The request was not valid.",
    "not_verified": "The patient has not been verified yet, so personal records and actions are unavailable.",
    "patient_mismatch": "That request is not for the verified patient.",
    "not_owner": GENERIC["not_found"],
    "tool_locked": "Booking changes are disabled for this conversation; staff will help.",
    "needs_confirmation": "Confirmation required: ask the patient to confirm this exact action, then repeat the call.",
}
PUBLIC_ERROR = {"not_owner": "not_found"}

# A confirmation is a short message made ONLY of yes-words and polite filler, with at least one yes-word.
# Anything else ("ok book Dr. Mehta tomorrow", "please cancel", "haan lekin ...") is not a confirmation.
AFFIRM = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "confirm", "confirmed", "haan", "han", "ha", "haa",
          "ji", "jee", "theek", "thik", "sahi", "bilkul", "pakka", "done",
          "हाँ", "हां", "हा", "जी", "ठीक", "सही", "बिल्कुल", "पक्का"}
FILLER = {"please", "pls", "plz", "kar", "karo", "kardo", "do", "de", "dena", "dijiye", "dijie", "kijiye",
          "kariye", "is", "hai", "hain", "to", "toh", "hi", "that", "it", "sounds", "good", "great", "fine",
          "perfect", "go", "ahead", "thanks", "thank", "you", "sir", "madam", "mam", "maam",
          "है", "हैं", "तो", "ही", "कर", "करो", "कीजिए", "करिए", "दीजिए", "दो", "दें", "धन्यवाद", "शुक्रिया",
          "सर", "मैम"}


def is_affirmation(text):
    """True for a short, unqualified yes in English, Hindi or Hinglish ("haan ji", "ठीक है", "yes please")."""
    toks = re.findall(r"[\wऀ-ॿ]+", (text or "").lower().replace("'", "").replace("’", ""))
    if not toks or len(toks) > 8:
        return False
    return any(t in AFFIRM for t in toks) and all(t in AFFIRM or t in FILLER for t in toks)


def action_key(name, args):
    return name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)


def describe_action(clinic: Clinic, name, args):
    """Human-readable summary of a state-changing call, generated from the clinic data, not by the LLM."""
    def when(slot):
        d, t = slot.start.split("T")
        return f"{d} at {t}"

    if name == "book_appointment":
        s, d = clinic.slots.get(args["slot_id"]), clinic.doctors.get(args["doctor_id"])
        p = clinic.get_patient(args["patient_id"])
        if s and d and p:
            return f"Book {d.name} ({d.specialty}) on {when(s)} for {p.name}"
        return f"Book slot {args['slot_id']} with doctor {args['doctor_id']}"
    if name == "cancel_appointment":
        a = clinic.get_appointment(args["appointment_id"])
        if a:
            return f"Cancel the appointment with {clinic.doctors[a.doctor_id].name} on {when(clinic.slots[a.slot_id])}"
        return f"Cancel appointment {args['appointment_id']}"
    return name


@dataclass
class Session:
    """Per-conversation state kept by the harness. The caller phone comes from the chat channel, never from
    the LLM; the verified patient id is set only by verify()."""
    caller_phone: str
    patient_id: Optional[str] = None
    booking_locked: bool = False
    pending: Optional[str] = None
    pending_summary: Optional[str] = None
    confirmed: Optional[str] = None
    last_user_affirmed: bool = False
    turn: int = 0

    def verify(self, clinic: Clinic, claimed_patient_id):
        """Verified if the caller's phone is registered on the claimed patient (own phone or a proxy)."""
        p = clinic.get_patient(claimed_patient_id)
        if p is not None and self.caller_phone in p.registered_phones:
            self.patient_id = claimed_patient_id
            return True
        return False

    def lock_booking(self):
        self.booking_locked = True
        self.pending = self.pending_summary = self.confirmed = None

    def on_user_message(self, text):
        """Call once per patient message, before the orchestrator runs."""
        self.turn += 1
        self.last_user_affirmed = is_affirmation(text)
        if self.pending and self.last_user_affirmed:
            self.confirmed, self.pending, self.pending_summary = self.pending, None, None
        else:
            self.pending = self.pending_summary = self.confirmed = None


@dataclass
class Verdict:
    allowed: bool
    reason: str = "ok"

    @property
    def user_message(self):
        return USER_MESSAGES.get(self.reason, "")


def evaluate(clinic: Clinic, session: Session, name, args) -> Verdict:
    if not isinstance(name, str) or name not in TOOLS:
        return Verdict(False, "unknown_tool")
    spec = TOOLS[name]
    if validate_args(name, args):
        return Verdict(False, "schema_invalid")
    if name == "lookup_policy":                     # general clinic info: no identity needed
        return Verdict(True)
    if session.patient_id is None:
        return Verdict(False, "not_verified")
    if "patient_id" in args and args["patient_id"] != session.patient_id:
        return Verdict(False, "patient_mismatch")
    if name == "cancel_appointment":
        a = clinic.get_appointment(args["appointment_id"])
        if a is None or a.patient_id != session.patient_id:     # nonexistent == not yours: same answer
            return Verdict(False, "not_owner")
    if spec.mutating:
        if session.booking_locked:
            return Verdict(False, "tool_locked")
        if session.confirmed != action_key(name, args):
            return Verdict(False, "needs_confirmation")
    return Verdict(True)


@dataclass
class Executor:
    """Runs tool calls for one episode. guard_on=False executes directly (the 'guard off' arms) but still
    records what the guard would have said. `log` is the clinic's audit list (one list, no duplicate)."""
    clinic: Clinic
    session: Session
    guard_on: bool = True
    log: list = field(init=False, repr=False)

    def __post_init__(self):
        self.log = self.clinic.audit

    def call(self, name, args) -> ToolResult:
        verdict = evaluate(self.clinic, self.session, name, args)
        spec = TOOLS.get(name) if isinstance(name, str) else None
        mutating = bool(spec and spec.mutating)
        summary = None
        if verdict.reason == "needs_confirmation":                # bookkeeping only, not enforcement
            summary = describe_action(self.clinic, name, args)
            self.session.pending = action_key(name, args)
            self.session.pending_summary = summary
        executed = verdict.allowed or not self.guard_on
        if executed:
            result = run_tool(self.clinic, name, args)
            if result.ok and mutating:
                self.session.confirmed = None                      # one confirmation, one action
        else:
            msg, data = verdict.user_message, None
            if summary:
                msg, data = f"{msg} Action to confirm: {summary}.", {"summary": summary}
            result = ToolResult(False, data=data, error=PUBLIC_ERROR.get(verdict.reason, verdict.reason),
                                message=msg)
        violation = (not verdict.allowed) and executed
        self.log.append({
            "turn": self.session.turn, "tool": name if isinstance(name, str) else repr(name)[:40],
            "args": args, "mutating": mutating, "guard_on": self.guard_on, "verdict": verdict.reason,
            "executed": executed, "ok": result.ok, "session_patient": self.session.patient_id,
            "user_affirmed_last_turn": self.session.last_user_affirmed,
            # diagnostics of the guard's own rules (NOT the reported metric; see d4_score.py)
            "unapproved": violation and mutating and result.ok,
            "leak": violation and result.ok and verdict.reason in IDENTITY_REASONS})
        return result
