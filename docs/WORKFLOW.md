# Local <-> Kaggle workflow
1. Write + test code locally (CPU, tiny samples).
2. Zip `src/` + needed `data/` files -> upload as a Kaggle Dataset (version it: v1, v2...).
3. Kaggle notebook: attach the dataset, run a script from `kaggle/`, write outputs to `/kaggle/working/`.
4. Download outputs -> put in `outputs/raw/<experiment_name>/`.
5. Small metric files -> copy to `outputs/results/` and commit.
6. Every experiment gets a row in `docs/EXPERIMENT_LOG.md`.

Rules: fixed seeds (13, 42, 2026); every Kaggle script reads a config in `configs/`
and saves `config + git-commit/dataset-version + metrics.json` next to its outputs.
