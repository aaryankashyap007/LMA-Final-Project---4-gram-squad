"""Shared labels and the structured case specification (D3)."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional

LABELS = ["EMERGENCY", "URGENT", "ROUTINE", "INSUFFICIENT"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
SCRIPTS = ["en", "deva", "hinglish"]
CATEGORIES = [
    "airway_breathing_consciousness", "heart_brain", "bleeding_injury_poisoning",
    "maternal", "newborn_child", "general_medical", "mental_health_crisis",
]


@dataclass
class CaseSpec:
    case_id: str
    category: str
    label: str                       # gold urgency label
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: List[str] = field(default_factory=list)
    duration: Optional[str] = None
    severity: Optional[str] = None
    red_flag_cues: List[str] = field(default_factory=list)  # cue checklist
    pair_id: Optional[str] = None    # minimal-pair group
    split: str = "train"

    def to_dict(self):
        return asdict(self)
