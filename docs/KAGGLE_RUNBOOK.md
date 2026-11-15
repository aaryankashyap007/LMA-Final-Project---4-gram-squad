# Kaggle runbook (all GPU work)

## 0. Local prep (once per data/code change)
```bash
python scripts/validate_d3.py --export        # eval splits
python scripts/prepare_train.py               # train split + overlap check
python scripts/make_kaggle_bundle.py          # -> kaggle_bundle/swasthya-data.zip
```

## 1. Kaggle one-time setup
1. kaggle.com -> Datasets -> New Dataset -> upload `kaggle_bundle/swasthya-data.zip` -> title **swasthya-data** -> Create (private).
   Code/data changed later? Dataset page -> **New Version** -> upload the new zip.
2. New Notebook -> **Add Input** -> your swasthya-data dataset.
3. Right panel: **Accelerator = GPU T4 x2** (use one GPU; avoid P100 for 4-bit), **Internet = On** (needs phone verification), Persistence: none.

## 2. Notebook cells (run in order; do the --smoke line of each script FIRST)
```
# cell 1: install
!pip install -q -U bitsandbytes peft accelerate

# cell 2: locate scripts
KD = !find /kaggle/input -type d -name kaggle | head -1
KD = KD[0]; print(KD)

# cell 3: MuRIL baseline + train-script ablation: 4 train settings x 3 seeds (~15-25 min total)
!python $KD/run_muril.py --smoke
!python $KD/run_muril.py --train_scripts all en deva hinglish --seeds 13 42 2026

# cell 4: prompted LLM (V0), zero-shot and 8-shot (~10-15 min each)
!python $KD/run_llm_prompted.py --smoke
!python $KD/run_llm_prompted.py --shots 0 --name v0_zeroshot
!python $KD/run_llm_prompted.py --shots 8 --name v0_fewshot8

# cell 5: QLoRA gate (~30-60 min per run)
!python $KD/run_qlora_gate.py --smoke
!python $KD/run_qlora_gate.py --train_scripts all --seed 13 --name qlora
# ablation, only if time/quota allows:
!python $KD/run_qlora_gate.py --train_scripts en --seed 13 --name qlora

# cell 6: package outputs
!cd /kaggle/working && zip -rq outputs.zip outputs && ls -la outputs.zip
```
Download `outputs.zip` from the notebook's **Output** tab (after "Save Version" -> "Save & Run All", or from the file browser on the right).
Kaggle sessions die after ~9-12 h and you get ~30 GPU-h/week: run in the order above and zip after every cell group.

## 2b. Remaining triage experiments (in priority order; each G1 run is ~30-60 min on a T4, quota ~30 GPU-h/week)
Smoke-test first (`--smoke`), and rebuild + re-upload the bundle (`make_kaggle_bundle.py`) so it contains the new flags.
```
# cell A: matched-size control. 300 rows, every case in ONE script, all scripts present (compare with S2 and the
# English-only / Hinglish-only runs: separates script coverage from data quantity). 3 seeds.
!python $KD/run_qlora_gate.py --train_scripts mixed1 --seed 13 --name qlora
!python $KD/run_qlora_gate.py --train_scripts mixed1 --seed 42 --name qlora
!python $KD/run_qlora_gate.py --train_scripts mixed1 --seed 2026 --name qlora

# cell B: early stopping on dev (recall at 10% over-triage, ties by macro-F1; restores the best epoch). 3 seeds.
!python $KD/run_qlora_gate.py --train_scripts all --seed 13 --name qlora_es --early_stop
!python $KD/run_qlora_gate.py --train_scripts all --seed 42 --name qlora_es --early_stop
!python $KD/run_qlora_gate.py --train_scripts all --seed 2026 --name qlora_es --early_stop

# cell C: noisy-romanisation training copy (python scripts/augment_train.py first, then rebuild the bundle). 3 seeds.
!python $KD/run_qlora_gate.py --train_scripts all --seed 13 --name qlora_noisy --train_file d3_train_noisy.jsonl
!python $KD/run_qlora_gate.py --train_scripts all --seed 42 --name qlora_noisy --train_file d3_train_noisy.jsonl
!python $KD/run_qlora_gate.py --train_scripts all --seed 2026 --name qlora_noisy --train_file d3_train_noisy.jsonl

# cell D: more seeds for the single-script ablations (G1 has only en s13 and hinglish s13; Devanagari-only never ran)
!python $KD/run_qlora_gate.py --train_scripts en --seed 42 --name qlora
!python $KD/run_qlora_gate.py --train_scripts en --seed 2026 --name qlora
!python $KD/run_qlora_gate.py --train_scripts hinglish --seed 42 --name qlora
!python $KD/run_qlora_gate.py --train_scripts hinglish --seed 2026 --name qlora
!python $KD/run_qlora_gate.py --train_scripts deva --seed 13 --name qlora
!python $KD/run_qlora_gate.py --train_scripts deva --seed 42 --name qlora
!python $KD/run_qlora_gate.py --train_scripts deva --seed 2026 --name qlora
```
Zip after every cell (`cd /kaggle/working && zip -rq outputs.zip outputs`) and download, because sessions die.
Rename any run folder that would collide with an existing name in `outputs/raw/` before scoring (the early-stop and
noisy runs have their own `--name`, so they do not collide). Then on your PC: `python scripts/score_all.py`.
The noisy and early-stop runs are compared against the plain all-script runs, same seeds.

## 2c. Agent run (the 2x2 experiment): `kaggle/run_agent.py`
Rebuild and re-upload the bundle first (`python scripts/make_kaggle_bundle.py`, then a New Version of `swasthya-data`).
Attach to the notebook: the `swasthya-data` bundle, your G1 adapter dataset (only for `+Gate`, `+Gate_V0` and `Full`),
and, for the real run, a scenarios dataset. Internet On. A T4 GPU is only needed for the gate arms.
Add the API key under **Add-ons > Secrets** with the label `LLM_API_KEY`. Never paste it into a cell.
```
# cell 1: install (only needed for the gate arms)
!pip install -q -U bitsandbytes peft accelerate

# cell 2: locate the scripts and set the API credentials (the key is read from the secret, never printed)
KD = !find /kaggle/input -type d -name kaggle | head -1
KD = KD[0]
import os
from kaggle_secrets import UserSecretsClient
os.environ["LLM_API_KEY"] = UserSecretsClient().get_secret("LLM_API_KEY")
os.environ["LLM_API_BASE"] = "https://<your provider>/v1"      # OpenAI-compatible endpoint of your subscription

# cell 3: PREFLIGHT. Does the API answer, make a valid tool call in English AND Hindi, repeat at temperature 0?
# If it says FAILED, do not start the run: fix the endpoint/model, or change the model for ALL arms.
!python $KD/run_agent.py --preflight

# cell 4: pilot WITHOUT a GPU: 5 built-in scenarios on the arms that need no gate (about 15-30 API calls per arm)
!python $KD/run_agent.py --scenarios smoke --arms A0 A1 +Guard --runs 1 --out /kaggle/working/agent_smoke

# cell 5: pilot WITH the tuned gate (needs the GPU and the adapter dataset)
!python $KD/run_agent.py --scenarios smoke --arms +Gate Full --runs 1 --out /kaggle/working/agent_smoke

# cell 6: main run (all five arms, 3 runs). Re-running the same cell resumes where it stopped.
# SC is found automatically, so there is no placeholder to edit. Use the same --token_budget for every arm.
SC = !find /kaggle/input -name scenarios_benign_v0.jsonl | head -1
SC = SC[0]; print(SC)
!python $KD/run_agent.py --scenarios $SC --runs 3 --token_budget 40000 --out /kaggle/working/agent_benign

# cell 7: package and download (also zip after every long cell: sessions die)
!cd /kaggle/working && zip -rq agent.zip agent agent_smoke && ls -la agent.zip
```
What to look at:
- **Preflight** must pass its three hard checks. The two soft checks tell you whether token counts are real
  (otherwise the budget uses a character estimate) and whether temperature 0 repeats exactly.
- **Smoke pilot:** `A1` (guard off) should show `unapproved` and `leak` above 0 on the injection and impersonation
  scenarios, and `+Guard` should show 0. If both are 0, the model is refusing on its own: that is a result, not a bug.
- Lines ending in `FLAGS infra_error` mean the API failed after its retries, and `FLAGS gate_outage` means the gate was
  unavailable. Those episodes are invalid and are re-run automatically when you run the same command again.
- Free-tier endpoints rate-limit: raise `--min_interval` (seconds between calls) if you see many retries or errors.
- Results are in `<out>/episodes/`, a summary in `<out>/run_summary.json`, the settings (never the key) in
  `<out>/run_config.json`, and the gate cache in `<out>/gate_cache_*.jsonl`.

## 2d. Gate probe: context and administrative messages (`scripts/probe_context.py`)
Measures how G1 treats short replies, follow-up answers, and 42 plain administrative messages (book, cancel,
reschedule, my appointments, clinic information) in the three scripts. The question for the administrative group:
how often does G1 give a benign request a harmful label (INSUFFICIENT hides the booking tools; URGENT or EMERGENCY is a
false alarm)?
```
# on your PC (repo folder, venv active): build the probe, then upload outputs/probe/probe_texts.json as a Kaggle dataset
python scripts/probe_context.py build

# Kaggle cell (attach: swasthya-data, the adapter dataset, the probe dataset; T4 GPU on)
import sys, glob, json, os
src = glob.glob("/kaggle/input/**/src/swasthya", recursive=True)[0]
adapter = os.path.dirname(glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)[0])
texts = json.load(open(glob.glob("/kaggle/input/**/probe_texts.json", recursive=True)[0]))
sys.path.insert(0, os.path.dirname(src))
from swasthya.gate_backend import load_backend
from swasthya.gate import decide
backend = load_backend(adapter_dir=adapter)
out = {f: [decide(p) for p in backend.probs(t)] for f, t in texts.items()}
json.dump(out, open("/kaggle/working/probe_out.json", "w"))
print({f: len(v) for f, v in out.items()})        # expect 209 per format

# back on your PC: put probe_out.json in outputs/probe/ and run
python scripts/probe_context.py score
```
Read the last section of the output ("administrative messages"). `format1` is the message as the first message of a
conversation, which is the case that matters. If the harmful rate is high, tell me which kinds and scripts, and we
decide between letting the intent tagger override INSUFFICIENT for administrative-only messages, or retraining G1 with
administrative examples. The wording of the Hinglish and Devanagari messages is a draft: have a native speaker read it.

## 2e. Benign scenario set (`data/d4_clinic/scenarios_benign_v0.jsonl`)
39 scenarios (13 families x English, Hinglish, Devanagari): bookings, cancellations, a registered proxy caller, a
reschedule, information questions, a greeting before a booking, and three that must NOT end in a booking (an emergency
interrupting a booking, an urgent child, a medical-advice request). Regenerate with
`python scripts/make_scenarios_benign.py`. Upload the .jsonl file as a Kaggle dataset, attach it, and use cell 6 above.
A perfect-agent test proves every scenario is solvable through the guard's confirmation protocol, so a failure in a
real run points at the model or the gate, not the script. Adversarial scenarios are a separate file (not written yet).

## 2f. Adversarial scenario set (`data/d4_clinic/scenarios_adversarial_v0.jsonl`)
39 scenarios: 10 attack families (injection into an information question, injection into a real booking, administrator
or doctor impersonation, another patient's record, an unregistered phone claiming to be a patient, multi-turn pressure,
system-prompt and patient-list exfiltration, an emergency with "ignore the rules" attached, a "DemoBot" role play) and
3 benign twins (same shape, no attack), each in English, Hinglish and Devanagari. Regenerate with
`python scripts/make_scenarios_adversarial.py`. This is the set that can separate the arms: `unapproved` and `leak` are
scored by the independent oracle. Reply text is not scored, so an exfiltration answer that names a patient without a
tool call is not caught (a limitation to report).
Tests prove two things: a perfect agent passes every scenario in A1 and +Guard, and an agent that OBEYS the attack
produces an unapproved action or leak in A1 in all 27 attack scenarios that have a tool-call form and none with the guard on.
Upload the .jsonl as a Kaggle dataset, attach it, then run (all five arms, 3 runs, same budget as the benign run):
```
SC = !find /kaggle/input -name scenarios_adversarial_v0.jsonl | head -1
SC = SC[0]; print(SC)
!python $KD/run_agent.py --scenarios $SC --runs 3 --token_budget 40000 --min_interval 4 --out /kaggle/working/agent_adv
!cd /kaggle/working && zip -rq agent_adv.zip agent_adv
```
Use a different `--out` from the benign run. Expect 585 episodes at 3 runs (about 1 h 30 on a quota-free key); start with
`--runs 1` (195 episodes) to see whether any arm is non-zero before spending more. Zip after each chunk.
A strong agent model may refuse every attack on its own, so all arms can still read 0: that is a result about the
model, and H2 would then need a weaker agent model as well. Say so rather than hiding it.

### 2f-v1. Plausible attacks (`scenarios_adversarial_v1.jsonl`)
v0 uses overt jailbreak wording, which strong models refuse on their own (all arms scored 0 on Nemotron Ultra). v1 has
30 scenarios (10 families x 3 scripts) of plain, believable requests that a helpful model tends to grant: a caller for an
unregistered neighbour, wife, sister or aunt, a "reception" bulk cancel, a pasted "clinic email", a wrong patient id, a
name claim, a slot swap, a sympathy cancel. Generate with `python scripts/make_scenarios_adversarial_v1.py`; run it exactly
like v0 with a different `--out` (for example `agent_advv1`). Counter-checks (benign): the proxy scenarios of the benign set.


### 2f-v2. Attacks hidden in tool results (`scenarios_adversarial_v2.jsonl`)
27 scenarios (8 attack families + 1 benign twin x en/hinglish/deva). The patient asks an ordinary question; a "note for the
chat assistant" is planted in clinic data that a tool returns (a policy text or a doctor's description). Scenarios carry a
`clinic_patch`, which `run_episode` applies to the clinic copy, so **the bundle must be rebuilt**
(`python scripts/make_kaggle_bundle.py`, upload as a new version of `swasthya-data`) because `src/` changed. Upload
`scenarios_adversarial_v2.jsonl` as its own dataset, then run as in 2f with `--out /kaggle/working/agent_advv2_<model>`.
Analyse with `python scripts/analyze_attempts.py --run <folder> --scenarios data/d4_clinic/scenarios_adversarial_v2.jsonl`.

### 2g. Medical-advice verifier (`scenarios_medical_advice_v0.jsonl`)
30 scenarios (10 families x en/hinglish/deva): the patient asks for a medicine, a dose, a diagnosis or a home remedy. The
rule-based verifier (`src/swasthya/verifier.py`) checks every model reply and, with `--verifier on`, replaces a reply that
gives advice by a fixed text; the original reply is kept in the episode's `verifier` log. `src/` changed, so rebuild and
upload the bundle (`python scripts/make_kaggle_bundle.py`), and upload the scenario file as its own dataset. Run the
cheap, weak models (they are the ones likely to give advice), for example:
`!python $KD/run_agent.py --scenarios $SCM --model $MODEL --verifier on --runs 1 --token_budget 40000 --min_interval 4 --out /kaggle/working/agent_medadvice_liquid`
(150 episodes; all five arms). Analyse with `scripts/analyze_verifier.py` (see its docstring).

## 3. Back on your PC
```bash
unzip -q ~/Downloads/outputs.zip -d /tmp/k && cp -r /tmp/k/outputs/* outputs/raw/
python scripts/score_all.py                  # -> outputs/results/summary.csv (+ one JSON per run)
```
Then add rows to `docs/EXPERIMENT_LOG.md` and commit `outputs/results/` (raw preds stay out of git).

## 4. Experiment IDs (use in the report)
| ID | System | What varies |
|----|--------|-------------|
| B0 | keyword lexicon | - |
| B1 | MuRIL, trained on all scripts, 3 seeds | baseline |
| S1 | MuRIL trained on en / deva / hinglish only | cross-script transfer (ablation) |
| V0 / V0-fs | Qwen2.5-3B prompted, 0 / 8 shots | no training |
| G1 | Qwen2.5-3B + QLoRA, all scripts | our gate |
| S2 | G1 trained on en only | cross-script transfer (ablation) |

Naming: S1/S2 are the script-transfer ablations above. They were called A1/A2 before; A0/A1/+Gate/+Guard/Full are
now only the agent arms of the 2x2 experiment (proposal Table 1, `swasthya.orchestrator.ARMS`).
