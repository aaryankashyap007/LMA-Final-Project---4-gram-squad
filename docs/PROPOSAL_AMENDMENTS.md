# Proposal amendments

The proposal PDF is not edited here. Apply these to its LaTeX source so the text matches what was built.

## 1. The gate predicts four urgency labels only (no intent, no red-flag tags)

**Decision.** G1 (and the prompted gate V0) output one of `EMERGENCY, URGENT, ROUTINE, INSUFFICIENT`. They do not
predict intent or red-flag tags. The proposal text is changed to match, instead of adding heads to G1, because:

- D3 has cue lists for parity checking but no intent or red-flag label set mapped for training. Adding the heads
  means designing and annotating that label set and retraining every gate seed.
- Intent is only needed to choose a tool, and the orchestrator already does that.
- Nothing downstream reads red-flag tags. The staff hand-off carries the urgency, the patient id, the turn and the
  patient's own words (quoted as untrusted data), which is what staff need.

**Edits.**

- Section 4.1, first sentence, replace
  "which predicts one of four urgency states (EMERGENCY, URGENT, ROUTINE, INSUFFICIENT) plus intent and red-flag tags"
  with
  "which predicts one of four urgency states (EMERGENCY, URGENT, ROUTINE, INSUFFICIENT)".
- Figure 1, safety-gate box: "urgency + red flags" becomes "urgency (4 classes)".
- Section 4.1 hand-off wording: "hands off to staff" becomes "hands off to staff with the urgency level, the patient id
  and the patient's message (quoted as untrusted text)".
- Section 13 deliverables and the Section 4 capability table need no change.
- Add to Section 10, Tier 3 / future work: "Multi-task gate heads for intent and red-flag tags".

## 2. Names of the experiment IDs

The runbook's script-transfer ablations were called A1/A2, which clashed with the proposal's agent arms A0/A1
(Table 1). They are now S1 (MuRIL, single training script) and S2 (G1, English only). A0, A1, +Gate, +Guard and Full
mean only the agent arms. The code also has `+Gate_V0` (a prompted Qwen gate with the guard off), added so that
"separating the decision from the LLM" can be told apart from "fine-tuning the gate".
