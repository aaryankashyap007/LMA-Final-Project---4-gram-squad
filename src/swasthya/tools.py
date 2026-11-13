"""The four clinic tools the orchestrator may call, with strict argument schemas.

Tools do business logic only (slot free? appointment exists?). They never decide who is allowed to call
them: identity, locking and confirmation belong to the action guard (guard.py).

Only `ToolResult.message` and `.data` may be shown to the LLM. Error codes are coarse "public" codes, and the
messages never echo caller-supplied text or reveal other patients' data.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .clinic import Clinic, ClinicError

ID_PATTERNS = {"patient_id": r"P[0-9]{3}", "doctor_id": r"D[0-9]{2}", "slot_id": r"S[0-9]{3}",
               "appointment_id": r"A[0-9]{3}"}         # [0-9], not \d: \d also matches Devanagari digits


@dataclass(frozen=True)
class ToolSpec:
    name: str
    mutating: bool
    args: Dict[str, str]        # arg name -> "id" (format in ID_PATTERNS) or "text" (free string)


TOOLS = {
    "lookup_appointments": ToolSpec("lookup_appointments", False, {"patient_id": "id"}),
    "book_appointment": ToolSpec("book_appointment", True,
                                 {"patient_id": "id", "doctor_id": "id", "slot_id": "id"}),
    "cancel_appointment": ToolSpec("cancel_appointment", True, {"appointment_id": "id"}),
    "lookup_policy": ToolSpec("lookup_policy", False, {"query": "text"}),
}


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""             # coarse public code
    message: str = ""           # what the orchestrator sees on failure


def validate_args(name, args):
    """Schema errors (empty = valid): known tool, exact argument set, types, id formats.
    Error texts never echo caller-supplied strings."""
    if not isinstance(name, str) or name not in TOOLS:
        return ["unknown tool"]
    spec = TOOLS[name]
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    errs: List[str] = []
    errs += [f"missing argument {k}" for k in spec.args if k not in args]
    if any(k not in spec.args for k in args):
        errs.append("unexpected argument")
    for k, kind in spec.args.items():
        if k not in args:
            continue
        v = args[k]
        if not isinstance(v, str):
            errs.append(f"{k} must be a string")
        elif kind == "id" and not re.fullmatch(ID_PATTERNS[k], v):
            errs.append(f"{k} has a bad format")
        elif kind == "text" and not (1 <= len(v.strip()) <= 300):
            errs.append(f"{k} must be 1-300 characters")
    return errs


def _tokens(text):
    """Crude stems (first 5 letters) so 'cancel' matches 'cancelled'. Placeholder for BGE-M3 retrieval."""
    return {t[:5] for t in re.findall(r"\w+", text.lower())}


STOP = {t[:5] for t in ("the", "a", "an", "is", "are", "to", "of", "and", "or", "at", "for", "in", "on", "can",
                        "be", "by", "what", "how", "do", "i", "my", "me", "it", "up", "every")}


def _policy_lookup(clinic: Clinic, query, k=2):
    """A policy matches if it shares 2+ content words with the query, or covers every content word of a very
    short query; otherwise abstain (one shared word like 'appointment' is not an answer)."""
    q = _tokens(query) - STOP
    scored = []
    for p in clinic.policies:
        overlap = len(q & (_tokens(p["text"]) - STOP))
        if overlap >= 2 or (overlap >= 1 and overlap == len(q)):
            scored.append((overlap, p["id"], p["text"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    if not scored:
        return {"snippets": [], "abstain": True}
    return {"snippets": [{"id": i, "text": t} for _, i, t in scored[:k]], "abstain": False}


GENERIC = {"bad_arguments": "The request was not valid. Check the arguments and try again.",
           "not_found": "That item could not be found.",
           "slot_taken": "That slot is no longer available.",
           "too_late": "This appointment can no longer be changed online; staff will help.",
           "patient_busy": "The patient already has an appointment at that time.",
           "failed": "The action could not be completed."}

PUBLIC = {"patient_not_found": "not_found", "slot_not_found": "not_found",
          "appointment_not_found": "not_found", "slot_taken": "slot_taken", "slot_in_past": "slot_taken",
          "too_late": "too_late", "patient_busy": "patient_busy",
          "doctor_mismatch": "failed", "already_cancelled": "failed"}


def run_tool(clinic: Clinic, name, args) -> ToolResult:
    """Execute a tool. Assumes nothing about who is calling; validates arguments itself."""
    errs = validate_args(name, args)
    if errs:
        return ToolResult(False, error="bad_arguments", message=GENERIC["bad_arguments"] + " " + "; ".join(errs))
    try:
        if name == "lookup_appointments":
            rows = [{"appointment_id": a.appointment_id, "doctor_id": a.doctor_id, "slot_id": a.slot_id,
                     "start": clinic.slots[a.slot_id].start}
                    for a in clinic.appointments_for(args["patient_id"])]
            return ToolResult(True, {"appointments": rows})
        if name == "book_appointment":
            a = clinic.book(args["patient_id"], args["doctor_id"], args["slot_id"])
            return ToolResult(True, {"appointment_id": a.appointment_id, "slot_id": a.slot_id,
                                     "start": clinic.slots[a.slot_id].start})
        if name == "cancel_appointment":
            a = clinic.cancel(args["appointment_id"])
            return ToolResult(True, {"appointment_id": a.appointment_id, "status": a.status})
        if name == "lookup_policy":
            return ToolResult(True, _policy_lookup(clinic, args["query"]))
    except ClinicError as e:
        code = PUBLIC.get(e.code, "failed")
        return ToolResult(False, error=code, message=GENERIC[code])
    return ToolResult(False, error="bad_arguments", message=GENERIC["bad_arguments"])
