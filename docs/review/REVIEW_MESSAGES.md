# Review messages

All items are synthetic: no real patient is described. The system is a research prototype, not a medical device.

## Native-speaker reviewer (Hindi / Hinglish)

> Hi! I am building a clinic chat assistant that understands English, Hindi and Hinglish, and I need a fluent speaker to
> check the wording. Attached is a spreadsheet (`review_native.csv`, open it in Excel or Google Sheets). Each row is a
> message a patient might send, written in English, Hinglish (Hindi in English letters) and Devanagari Hindi. Please
> tell me: (1) do the three versions mean the same thing, (2) does the Hinglish and the Devanagari sound like something
> a real person would write, and if not, how would you say it. There are about 100 rows; reading a sample of 50-60 is
> fine. It should take about an hour. Rows marked 'planted note' are texts that appear inside the clinic's own documents. The last rows are the Hindi replies the assistant gives in an emergency: please
> check they are clear and polite. Could you send it back by <date>? Thank you!

What to fill in per row: column G (Y/N same meaning), H and J (OK or CHANGE), I and K (your wording if CHANGE), L (comment).

## Clinician reviewer

> Hi! I am building a research prototype of a clinic front-desk chat assistant that sorts messages into emergency,
> urgent, routine or too vague. I would value 45-60 minutes of a clinician's judgement. Attached is `review_clinician.csv`.
> (1) Please read the four triage categories at the top and tell me if they are reasonable. (2) Please give your own label
> for each of the 40 short messages (column F), without looking anything up: EMERGENCY, URGENT, ROUTINE or INSUFFICIENT
> (too vague to judge). If torn, choose the more urgent one. (3) Please read the three fixed replies and tell me if
> they are safe, and answer the four questions at the end. All patients are imaginary. Could you send it back by <date>?
> Thank you!

What to fill in per row: column F (your label, or Y/N for the rubric and reply rows), G (comment).

## After the sheets come back
- Native speaker: apply the CHANGE rows to the scenario generators and the D3 text, regenerate the sets, and re-run what changed.
- Clinician: `python scripts/make_review_sheets.py --score docs/review/review_clinician_filled.csv` prints agreement and kappa
  against our labels, and which cases they disagree on. Report the agreement rate; do not silently relabel D3.
