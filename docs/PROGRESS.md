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
- [x] Adversarial v0 on Nemotron Nano (195 episodes): first real-model separation, A1 and +Gate 2 unapproved, guard arms 0
## In progress
## Not started
- [ ] `+Gate_V0` arm: run it or state that it was dropped
- [ ] Native-speaker review of Hinglish and Devanagari wording; clinician review of rubric and templates
- [ ] Adapter backup, report tables, push of `agent-lane`
## Challenges + mitigations (fill as they happen)
| Challenge | Impact | Mitigation |
|-----------|--------|-----------|
| Free API quota stopped the benign run at episode 91; a call hung for hours | lost time, flagged episodes | resumable runner skips finished episodes and redoes flagged ones; separate output folder per run |
| Kaggle restarts wipe the working folder | finished episodes lost | zip and download after each run; restore from an uploaded dataset |
| Guard arms refused "my father's appointment" because the prompt said the patient was verified | proxy scenarios failed in guard arms | prompt line for registered family callers; affected episodes re-run |
| Overt attacks were refused by every model, so all arms scored 0 | arms could not be separated | plausible v1 attacks, attempted-attack metric, worst-case agent, mutation testing |
| Even plausible v1 attacks rarely produced harmful attempts | real-model comparison stays one-sided | report worst case and mutation results as the guard evidence; consider attacks inside tool results |
