# Proposal amendments

The proposal PDF is not edited here. Apply these to its LaTeX source so the text matches what was built.

## 1. Gate output: urgency from G1, red-flag tags and intent from a deterministic tagger

**What was built.** The gate's output has three parts:

- **Urgency** (one of `EMERGENCY, URGENT, ROUTINE, INSUFFICIENT`): the fine-tuned QLoRA gate G1, as proposed.
- **Red-flag tags** (14 tags such as `breathing_difficulty`, `chest_pain_cardiac`, `suicidal_intent`) and **intents**
  (`book_appointment, cancel_appointment, reschedule, appointment_info, clinic_info, medical_advice, symptom_report,
  other`): a deterministic tagger (`swasthya.tags`), NOT learned by G1.

**Why the tags are not a G1 head.** D3's cue lists are 917 mostly one-off strings that mix symptoms, demographics,
durations and negations, so they cannot be used as a label set for a classifier head. A head would need a new
clinician-defined tag set, new annotation and a retrain of every seed. The tagger is multilingual (English, Devanagari,
romanised Hindi, code-mixed), handles negation ("no chest pain", "seene mein dard nahi hai") and spelling variants
("nhi", "sans"). It is checked against D3 cue-derived tags, which is a proxy and not clinician truth
(`scripts/eval_tagger.py`, results in `docs/TAGGER_EVAL.md`). Intents have no labelled data, so no accuracy is claimed.

**Measured agreement with cue-derived tags (test split, 315 renderings; train-split misses guided the lexicon).**
Precision is mostly 0.8-1.0, recall is 0.4-1.0 depending on the tag: weakest on `stroke_signs` (0.42),
`suicidal_intent` (0.44) and `pregnancy_danger` (0.44), and 74% of EMERGENCY renderings carry at least one tag. Treat
the tags as supporting information for staff, never as a safety decision. Nothing in the pipeline acts on them: they
are recorded in the transcript, the gate log and the staff hand-off. A medical-advice hand-off from the
`medical_advice` intent is NOT enabled; it would need a clinician-reviewed template.

**Edits to the proposal text.**

- Section 4.1, first sentence: "Every turn is normalised (Unicode clean-up and script detection) and passed through
  the safety gate, which predicts one of four urgency states (EMERGENCY, URGENT, ROUTINE, INSUFFICIENT); a
  deterministic tagger adds red-flag tags and intent."
- Figure 1, safety-gate box: "urgency + red flags" stays, with the footnote "urgency from the tuned gate, red flags
  from a rule-based tagger".
- Section 4 capability table, fine-tuning row: say that the tuned gate predicts urgency only.
- Section 4.1 hand-off wording: the hand-off carries the urgency level, the patient id, the script, the red-flag tags
  seen in the conversation, the intent, and the patient's message quoted as untrusted text.
- Section 10, Tier 3 / future work: "Learned red-flag and intent heads (multi-task gate), with a clinician-defined tag
  set".
- Limitations: "red-flag tags and intents come from an unvalidated rule-based tagger; tag accuracy was checked only
  against cue-derived labels."

## 2. Normaliser: Unicode and script

The Figure 1 normaliser box now does what it says (`swasthya.normaliser`):

- **Unicode (model-facing, conservative):** NFKC, zero-width and control characters removed (they can hide text),
  Devanagari and Arabic-Indic digits to ASCII, typographic quotes to ASCII, spaces collapsed within each line. It
  changes none of the 1,350 D3 texts, so G1's input is unaffected (tested).
- **Matching (aggressive):** lower-case, nukta removed, danda and punctuation to spaces, repeated letters shortened,
  romanised-Hindi spelling variants mapped to one form. Used by the tagger only.
- **Script detection:** `deva | hinglish | en | mixed | unknown`, 99.4-100% on D3 with no confusion between
  languages; the few errors are one- to two-word English messages reported as `unknown`.

## 3. Names of the experiment IDs

The runbook's script-transfer ablations were called A1/A2, which clashed with the proposal's agent arms A0/A1
(Table 1). They are now S1 (MuRIL, single training script) and S2 (G1, English only). A0, A1, +Gate, +Guard and Full
mean only the agent arms. The code also has `+Gate_V0` (a prompted Qwen gate with the guard off), added so that
"separating the decision from the LLM" can be told apart from "fine-tuning the gate".
