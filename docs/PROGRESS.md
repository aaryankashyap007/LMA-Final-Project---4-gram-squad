# Progress log (feeds the mid-submission)
## Done
- [x] Repo scaffold, D3 triage dataset (150 evaluation cases, 300 training cases, three scripts)
- [x] Triage systems B0, B1, V0, G1 with seeds; training-script and prompting ablations; matched over-triage analysis (H1)
- [x] G1 adapter retrained and saved (seed 13); administrative override for plain administrative messages; context probe
- [x] D4 clinic simulator, tools, deterministic action guard, orchestrator with six arms, independent episode scoring
- [x] Resumable agent runner, API preflight, confirmation simulator
- [x] Scenario sets: benign (39), adversarial v0 (39), adversarial v1 (30)
- [x] Agent runs on benign and adversarial v1 sets; attempted-attack analysis; scripted worst-case agent
- [x] Mutation testing of the guard (16 mutants, 6 probes)
- [x] Adversarial v2 (attacks hidden in tool results) on Nano and Liquid: 0 harmful attempts in 270 episodes
- [x] 2x2 statistics script and pooled analysis (no significant effect)
- [x] `+Gate_V0` arm run on the benign set and adversarial v0: all six arms have now run
- [x] Review sheets (native speaker, clinician) and a scorer for the clinician sheet; both sent
- [x] Native-speaker review returned: no wording changes requested (as reported by the author; the sheet is not stored here)
- [x] G1 adapter downloaded as a local backup
- [x] Single-script G1 runs finished: Devanagari-only (3 seeds), English-only and Hinglish-only (seeds 42 and 2026 added); training-script ablation now has three seeds per row
- [x] Medical-advice verifier built, tested and run live (refusal-aware v1 after the first live run); the author's own labels of `docs/review/verifier_live_replies.csv` would replace the assistant's provisional ones
- [x] Guideline cross-check of the rubric against the WHO/ICRC/MSF IITT (`docs/GUIDELINE_CROSSCHECK.md`); emergency-vs-not agreement 0.85, 2 possible under-triage cases reported
- [x] Proposal metrics: non-inferiority, per-script, tool-call accuracy, escalation timeliness, hand-off completeness (`scripts/analyze_agent_extras.py`)
- [x] Agent failure analysis, 8 cases (`docs/AGENT_FAILURE_ANALYSIS.md`)
- [x] Adversarial v0 on Nemotron Nano (195 episodes): first real-model separation, A1 and +Gate 2 unapproved, guard arms 0
## In progress
## Not started
- [ ] Not planned unless time allows: repeated runs and pass@3, RAG with BGE-M3 and its metrics, adaptive-intake metrics; Tier 2 items were cut by the week-4 go/no-go
- [ ] Report tables, figures and write-up (local commits only; the repo is not pushed)
## Decided not to do
- Clinician review of the rubric, labels and emergency replies: no clinician available. Stated as a limitation in the README and the report; no clinical-safety claim is made. The clinician sheet and scorer stay in `docs/review/` in case a reviewer is found.
## Challenges + mitigations (fill as they happen)
| Challenge | Impact | Mitigation |
|-----------|--------|-----------|
| Free API quota stopped the benign run at episode 91; a call hung for hours | lost time, flagged episodes | resumable runner skips finished episodes and redoes flagged ones; separate output folder per run |
| Kaggle restarts wipe the working folder | finished episodes lost | zip and download after each run; restore from an uploaded dataset |
| Guard arms refused "my father's appointment" because the prompt said the patient was verified | proxy scenarios failed in guard arms | prompt line for registered family callers; affected episodes re-run |
| Overt attacks were refused by every model, so all arms scored 0 | arms could not be separated | plausible v1 attacks, attempted-attack metric, worst-case agent, mutation testing |
| Even plausible v1 attacks rarely produced harmful attempts | real-model comparison stays one-sided | report worst case and mutation results as the guard evidence; consider attacks inside tool results |
