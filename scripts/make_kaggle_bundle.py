"""Bundle code + data for upload to Kaggle as ONE dataset.

Run from repo root:  python scripts/make_kaggle_bundle.py [--kaggle-user YOURNAME]
Creates kaggle_bundle/swasthya-data/ (folder) and kaggle_bundle/swasthya-data.zip (same content at zip top level).
Re-run + upload as a NEW VERSION whenever src/, kaggle/ or the data changes.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-user", default=None, help="writes dataset-metadata.json for the kaggle CLI")
    a = ap.parse_args()
    need = [ROOT / f"data/d3_triage/rendered/d3_{s}.jsonl" for s in ("train", "dev", "test")]
    miss = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if miss:
        print("Missing:", miss, "\n-> run validate_d3.py --export and prepare_train.py first")
        return 1
    out = ROOT / "kaggle_bundle" / "swasthya-data"
    if out.exists():
        shutil.rmtree(out)
    (out / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src/swasthya", out / "src/swasthya", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "kaggle", out / "kaggle", ignore=shutil.ignore_patterns("__pycache__", ".gitkeep"))
    (out / "data/d3_triage/rendered").mkdir(parents=True)
    for p in need:
        shutil.copy(p, out / "data/d3_triage/rendered" / p.name)
    if a.kaggle_user:
        (out / "dataset-metadata.json").write_text(json.dumps(
            {"title": "swasthya-data", "id": f"{a.kaggle_user}/swasthya-data",
             "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    zp = ROOT / "kaggle_bundle" / "swasthya-data.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out))
    print("bundle folder:", out.relative_to(ROOT))
    print("bundle zip   :", zp.relative_to(ROOT), f"({zp.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
