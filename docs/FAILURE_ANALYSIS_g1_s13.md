# Failure analysis: `outputs/raw/qlora_all_s13/preds_test.jsonl` on `data/d3_triage/rendered/d3_test.jsonl`

52 errors in 315 renderings. Tables are computed; the shortlist below is chosen by fixed rules (see `swasthya.failures.pick_candidates`) and still needs a human reading.

## Confusion matrix (rows = gold, columns = predicted)

| gold \ pred | EMERGENCY | URGENT | ROUTINE | INSUFFICIENT |
|---|---|---|---|---|
| **EMERGENCY** | 115 | 11 | 0 | 0 |
| **URGENT** | 17 | 48 | 3 | 1 |
| **ROUTINE** | 1 | 12 | 52 | 1 |
| **INSUFFICIENT** | 0 | 3 | 3 | 48 |

## Errors by type

- false_alarm: 18
- over_triage: 12
- missed_emergency: 11
- committed_on_vague: 6
- under_triage: 3
- wrongly_insufficient: 2

## Errors by script

- en: 7
- deva: 17
- hinglish: 28

## Errors by category

- newborn_child: 14
- general_medical: 9
- heart_brain: 9
- maternal: 9
- mental_health_crisis: 6
- bleeding_injury_poisoning: 4
- airway_breathing_consciousness: 1

## Cases by number of scripts predicted wrongly

(0 = right in all three scripts; 3 = wrong in all three; 1-2 = a script-specific failure)

- 0 wrong: 66 cases
- 1 wrong: 27 cases
- 2 wrong: 11 cases
- 3 wrong: 1 cases

## Minimal pairs that did not flip

- pair P07 (hinglish): cases C092 and C123 both predicted URGENT
- pair P08 (deva): cases C011 and C135 both predicted EMERGENCY
- pair P08 (hinglish): cases C011 and C135 both predicted EMERGENCY
- pair P15 (hinglish): cases C088 and C117 both predicted EMERGENCY
- pair P17 (deva): cases C006 and C074 both predicted URGENT
- pair P17 (hinglish): cases C006 and C074 both predicted URGENT
- pair P18 (en): cases C044 and C052 both predicted INSUFFICIENT

## Candidate cases (up to 8)

### 1. missed emergency (en): `C024|en`

- gold **EMERGENCY**, predicted **URGENT** (missed_emergency), category heart_brain
- probabilities: EME 0.07 / URG 0.93 / ROU 0.00 / INS 0.00
- same case in other scripts, predicted: en: URGENT, deva: EMERGENCY, hinglish: EMERGENCY
- cues: blurred_vision, difficulty_speaking, onset_20min, sudden
- text: Suddenly I cannot find the right words to speak and my vision has become blurry. It started 20 minutes ago.
- what went wrong: *(to be written after reading the case)*

### 2. missed emergency (deva): `C074|deva`

- gold **EMERGENCY**, predicted **URGENT** (missed_emergency), category newborn_child
- probabilities: EME 0.00 / URG 1.00 / ROU 0.00 / INS 0.00
- same case in other scripts, predicted: en: EMERGENCY, deva: URGENT, hinglish: URGENT
- cues: child_5yrs, diarrhoea, duration_2days, not_drinking, not_passing_urine, very_sleepy
- text: मेरे 5 साल के बेटे को 2 दिन से दस्त हो रहे हैं। वो कुछ पी नहीं रहा, आज पेशाब भी नहीं किया है और बहुत ज़्यादा सो रहा है।
- what went wrong: *(to be written after reading the case)*

### 3. missed emergency (hinglish): `C054|hinglish`

- gold **EMERGENCY**, predicted **URGENT** (missed_emergency), category newborn_child
- probabilities: EME 0.01 / URG 0.99 / ROU 0.00 / INS 0.00
- same case in other scripts, predicted: en: EMERGENCY, deva: EMERGENCY, hinglish: URGENT
- cues: newborn_4days, not_feeding, very_sleepy, yellow_skin
- text: meri 4 din ki bachchi poori peeli pad gayi hai. wo doodh nahi pi rahi aur bahut so rahi hai
- what went wrong: *(to be written after reading the case)*

### 4. false alarm: `C088|hinglish`

- gold **URGENT**, predicted **EMERGENCY** (false_alarm), category newborn_child
- probabilities: EME 1.00 / URG 0.00 / ROU 0.00 / INS 0.00
- same case in other scripts, predicted: en: URGENT, deva: URGENT, hinglish: EMERGENCY
- cues: child_2yrs, fever, no_seizure, playing
- text: meri 2 saal ki beti ko bukhar hai. usko daura nahi pada hai aur wo aaram se khel rahi hai
- what went wrong: *(to be written after reading the case)*

### 5. script-specific failure: `C043|hinglish`

- gold **EMERGENCY**, predicted **URGENT** (missed_emergency), category heart_brain
- probabilities: EME 0.02 / URG 0.98 / ROU 0.00 / INS 0.00
- same case in other scripts, predicted: en: EMERGENCY, deva: EMERGENCY, hinglish: URGENT
- cues: chest_pain, duration_40min, known_heart_disease, not_relieved_by_rest
- text: main dil ka mareez hoon aur 40 minute se mere seene mein dard hai. aaram karne se bhi kam nahi ho raha
- what went wrong: *(to be written after reading the case)*

### 6. minimal pair did not flip: `C135|hinglish`

- gold **URGENT**, predicted **EMERGENCY** (false_alarm), category bleeding_injury_poisoning
- probabilities: EME 1.00 / URG 0.00 / ROU 0.00 / INS 0.00
- same case in other scripts, predicted: en: URGENT, deva: EMERGENCY, hinglish: EMERGENCY
- cues: bleeding_stopped, caused_by_glass, deep_cut, leg, onset_1hour
- text: ek ghante pehle tute hue kaanch se meri taang gehri kat gayi thi. khoon bahna band ho gaya hai
- what went wrong: *(to be written after reading the case)*

### 7. under-triage URGENT->ROUTINE: `C039|deva`

- gold **URGENT**, predicted **ROUTINE** (under_triage), category heart_brain
- probabilities: EME 0.00 / URG 0.03 / ROU 0.97 / INS 0.00
- same case in other scripts, predicted: en: URGENT, deva: ROUTINE, hinglish: URGENT
- cues: duration_1week, irregular_heartbeat, no_chest_pain, on_and_off
- text: एक हफ़्ते से मेरी धड़कन रुक-रुक कर अनियमित हो जाती है। सीने में दर्द नहीं है।
- what went wrong: *(to be written after reading the case)*

### 8. INSUFFICIENT confusion: `C100|deva`

- gold **ROUTINE**, predicted **INSUFFICIENT** (wrongly_insufficient), category maternal
- probabilities: EME 0.00 / URG 0.00 / ROU 0.00 / INS 1.00
- same case in other scripts, predicted: en: ROUTINE, deva: INSUFFICIENT, hinglish: ROUTINE
- cues: family_planning_query
- text: मैं परिवार नियोजन के विकल्पों के बारे में डॉक्टर से बात करना चाहती हूँ।
- what went wrong: *(to be written after reading the case)*
