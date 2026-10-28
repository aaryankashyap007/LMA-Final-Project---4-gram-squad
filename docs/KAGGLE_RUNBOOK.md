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
| A1 | MuRIL trained on en / deva / hinglish only | cross-script transfer (ablation) |
| V0 / V0-fs | Qwen2.5-3B prompted, 0 / 8 shots | no training |
| G1 | Qwen2.5-3B + QLoRA, all scripts | our gate |
| A2 | G1 trained on en only | cross-script transfer (ablation) |
