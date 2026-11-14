"""D4 scenario: a scripted conversation plus the gold record used to score it (see d4_score.Gold)."""
from dataclasses import dataclass, field
from typing import List, Optional

from .d4_score import Gold


@dataclass
class Scenario:
    id: str
    caller_phone: str                       # supplied by the chat channel, never by the LLM
    claimed_patient_id: Optional[str]       # whom the caller says they are; verified against the phone
    user_turns: List[str]                   # scripted patient messages
    gold: Gold
    tags: List[str] = field(default_factory=list)       # e.g. ["proxy", "injection", "hinglish"]

    def to_dict(self):
        g = self.gold
        return {"id": self.id, "caller_phone": self.caller_phone, "claimed_patient_id": self.claimed_patient_id,
                "user_turns": list(self.user_turns), "tags": list(self.tags),
                "gold": {"patient_id": g.patient_id, "authorized": g.authorized,
                         "final_booked": sorted(list(t) for t in g.final_booked)}}

    @classmethod
    def from_dict(cls, d):
        g = d["gold"]
        gold = Gold(g["patient_id"], list(g.get("authorized", [])),
                    {tuple(t) for t in g.get("final_booked", [])})
        return cls(d["id"], d["caller_phone"], d.get("claimed_patient_id"), list(d["user_turns"]), gold,
                   list(d.get("tags", [])))
