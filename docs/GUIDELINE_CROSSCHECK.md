# Guideline cross-check of the triage rubric

**What this is.** The proposal's fallback for having no clinician is to "claim only agreement with a guideline-derived rubric". This document compares our four-level rubric and the 150 D3 evaluation labels with the WHO, ICRC and MSF **Interagency Integrated Triage Tool (IITT)**, the triage tool the proposal cites as reference [2]. **It is not clinician review and not a validation of our labels.** No clinical claim follows from it. The code is `scripts/guideline_crosscheck.py`; the numbers are in `outputs/results/guideline_crosscheck.json`; the cases that disagree, each with the authors' reading, are in `docs/review/iitt_disagreements.csv`.

**Source.** WHO, ICRC and MSF, *Interagency Integrated Triage Tool*, adult sheet (age 12 and over), paediatric sheet (under 12) and reference card, <https://www.who.int/tools/triage>, licence CC BY-NC-SA 3.0 IGO. The criteria are paraphrased in the script, not reproduced. The IITT has three tiers: red (seen immediately), yellow (seen soon) and green (can wait).

## How the comparison was made
- We encoded the IITT red and yellow criteria that a patient could describe in a text message (13 red, 14 yellow) as English keyword rules with a simple negation guard. Criteria that need a bedside measurement (heart rate, breathing rate, oxygen saturation, temperature, capillary refill, ECG, blood sugar) cannot be seen in a message and are left out.
- For each D3 case the rules give an IITT-derived tier: red if a red rule fires, else yellow if a yellow rule fires, else none. We map our EMERGENCY to red, URGENT to yellow and ROUTINE to none. INSUFFICIENT has no IITT counterpart and is excluded from agreement.
- The rules were written from the IITT sheets first and then widened to the wording of the D3 messages, so the agreement below is **optimistic**.

## Results (126 cases with a label that maps to an IITT tier)
| | IITT red | IITT yellow | IITT none |
|---|---|---|---|
| our EMERGENCY (60) | 43 | 10 | 7 |
| our URGENT (36) | 2 | 21 | 13 |
| our ROUTINE (30) | 0 | 5 | 25 |
| our INSUFFICIENT (24) | 0 | 1 | 23 |

- **Exact agreement of the tier:** 89 of 126 (0.71).
- **Emergency versus not** (our EMERGENCY against IITT red): 107 of 126 (0.85).
- **IITT red where our label is lower:** 2 cases, listed below. This is the direction that matters most for safety.
- **Our EMERGENCY where no IITT red rule fires:** 17 cases. Our rubric is more cautious than the IITT here, or the IITT has no criterion for the cue.
- No INSUFFICIENT case triggers a red rule.

## The 37 disagreements, grouped (authors' non-clinical reading)
| Group | Cases | Reading |
|---|---|---|
| Possible under-triage in our rubric | C053, C127 | C053: nine months pregnant with labour pains every 10 minutes; the IITT puts a pregnant patient in active labour in red, ours is URGENT. C127: pesticide on the hands, washed off, no symptoms; the IITT up-triages a high-risk exposure to red, ours is URGENT. |
| Mental-health crisis | C020, C045, C089, C102, C111, C119, C144 | The IITT is a physical-emergency tool with no self-harm or psychosis criterion. Our rubric treats suicidal intent as EMERGENCY. Scope difference, not an IITT view. |
| Stroke signs | C005, C024, C049, C123 | The IITT places an acute focal neurological complaint in yellow. Ours treats it as time-critical (EMERGENCY). |
| Our rubric more cautious than the IITT category | C004, C017, C022, C067, C078, C107, C120 | Head injury with drowsiness, cardiac-type symptoms, a drowsy diabetic, collapse with abdominal pain, thunderclap headache, breathless after childbirth. Consistent with our tie rule ("choose the more urgent level"). |
| Cue not visible to the rules or the tiers | C033, C039, C052, C061, C068, C081, C084, C088, C097, C122, C128, C132 | Symptoms, history or vital signs (fever, palpitations, a past fit, blood pressure) that the keyword rules or the IITT tiers do not capture. Mostly our URGENT against IITT none. |
| Rule false positive | C035, C036, C063 | A keyword fired on text that does not describe the current problem ("fainted once last year"). |
| IITT rule that does not fit | C086, C149 | Mild night wheeze in a known asthmatic asking for a check-up (the IITT lists wheezing as yellow); a three-month-old at a vaccination visit (the IITT puts every infant 8 days to 6 months in yellow). |

## Coverage of the IITT by D3
Every encoded red criterion is exercised by at least one D3 case, and 13 of 14 yellow criteria are. **Not exercised:** inability to pass urine. **Not encoded (cannot be seen in a message):** the vital-sign criteria, ECG, blood sugar, capillary refill, cold extremities. **Not encoded and not in D3:** threatened limb, sexual assault, testicular pain or priapism, open fracture or dislocation. The 150 cases are therefore a sample of the situations a general triage tool covers, not the whole.

## What can be said, and what cannot
- **Can say:** against a published triage tool, our labels agree on emergency versus not for 0.85 of the cases the rules can place, our rubric is on the cautious side where it differs (17 of 19 emergency-level differences), and two cases (C053, C127) look like possible under-triage that a clinician should look at.
- **Cannot say:** that the rubric, the labels or the emergency replies are clinically correct or safe. The IITT is a facility triage tool that also uses vital signs; our task is a text message. The mapping of our four levels to its three is an assumption. The keyword rules are crude and were tuned on D3's wording.
- **Not changed:** D3's labels. The two possible under-triage cases are reported, not relabelled; that decision needs a clinician.

## For the report
Quote this as: "As no clinician was available, we cross-checked our rubric against the WHO/ICRC/MSF Interagency Integrated Triage Tool using keyword rules for its text-observable criteria. The two schemes agreed on emergency versus non-emergency for 107 of 126 cases (0.85); differences were mostly mental-health and stroke presentations that the IITT scores differently or not at all. This is a consistency check and not a clinical validation."
