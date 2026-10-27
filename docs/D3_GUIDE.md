# D3 writing guide (human-written eval set)

Goal: ~100 cases total, 25 per person (initials prefix: AK, HN, PM, SM).
Each case = ONE row in a CSV, with the same clinical content written 3 ways.
Time: ~3 min/case.  Copy `data/d3_triage/human_cases/template.csv` to
`data/d3_triage/human_cases/<INITIALS>.csv` and fill it (LibreOffice/Sheets is fine; save as UTF-8 CSV).

## Columns
| column | what to write |
|---|---|
| case_id | `AK001`, `AK002`, ... (your initials + running number) |
| category | one of: airway_breathing_consciousness, heart_brain, bleeding_injury_poisoning, maternal, newborn_child, general_medical, mental_health_crisis |
| label | EMERGENCY / URGENT / ROUTINE / INSUFFICIENT |
| age | number (or blank) |
| sex | M / F / blank |
| cues | the clinical cues, `;`-separated, lowercase snake_case, e.g. `chest_pain;sweating;pain_left_arm;onset_30min` |
| en | English message, as a patient/relative would type it |
| deva | SAME content in Devanagari Hindi |
| hinglish | SAME content in romanised Hindi, hand-typed the way you would text (spelling variation OK) |
| pair_id | blank, or e.g. `AK-P1` for the two cases of a minimal pair |
| notes | optional |

## THE parity rule (most important)
Write the `cues` list FIRST. Then each of en / deva / hinglish must mention exactly those cues:
nothing missing, nothing extra (no extra symptom, no extra duration, no extra age hint).
Wording, spelling, politeness and filler may differ. Clinical content may not.

## Label rubric (planning version, to be clinician-reviewed)
- EMERGENCY: immediate danger to life -> staff alert now. e.g. cannot breathe / blue lips,
  chest pain with sweating or arm pain, face droop + slurred speech, unconscious/fits,
  heavy bleeding, poisoning, pregnant with heavy bleeding or no fetal movement + severe pain,
  newborn not feeding/limp/blue, suicidal intent with a plan.
- URGENT: needs a clinician the same day, not life threatening this minute. e.g. fever 3 days in a child
  that is drinking, deep cut that has stopped bleeding, painful urination with fever, persistent vomiting.
- ROUTINE: can wait for a normal appointment. e.g. mild cough 1 week, follow-up, BP check, old back pain.
- INSUFFICIENT: message too vague to triage ("tabiyat theek nahi hai", "doctor se milna hai, dikkat hai").

## Quotas per person (25 cases)
- 10 EMERGENCY, 6 URGENT, 5 ROUTINE, 4 INSUFFICIENT
- cover all 7 categories (3-4 each)
- 4 minimal pairs (= 8 of the 25 cases): two cases identical except ONE cue that flips the label.
  Example: `chest_pain;sweating;pain_left_arm` (EMERGENCY) vs `chest_pain;pain_on_pressing_chest_wall;after_gym` (ROUTINE).
  Give both rows the same pair_id.
- Write like real patients: short, typos, no medical jargon, sometimes a relative writing ("meri maa ko ...").
- Mix: some messages 1 line, some 3-4 lines; a few with an appointment request too ("kal ka appointment chahiye").
- All fake. No real names/phone numbers.

## Second-annotator parity check (after writing)
Swap CSVs with a teammate. They read each rendering SEPARATELY and fill:
`a2_present_en`, `a2_present_deva`, `a2_present_hinglish` = which cues from the `cues` column are present in that text (`;`-separated),
`a2_extra` = `yes` if ANY rendering has a clinical detail not in `cues`, else `no`.
Then send the file back. Validator reports the parity-pass rate per script.

## Checking your file
    python scripts/validate_d3.py              # validate all CSVs in human_cases/
    python scripts/validate_d3.py --export     # also write dev/test jsonl splits
