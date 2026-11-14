"""Resumable experiment runner: one json result per (scenario, arm, run).

Kaggle sessions die, so every finished episode is written to disk immediately (atomically), and a rerun skips
episodes that already have a result. Episodes that ended in an infrastructure error (LLM unreachable after
retries) or a gate outage (gate backend down or cache miss, so the turn failed closed) are rerun instead of
being scored as agent failures. Results that still carry such a flag after a run are not valid for analysis.
"""
import json
import os
from pathlib import Path

from .orchestrator import ARMS, run_episode

RERUN_FLAGS = ("infra_error", "gate_outage")


def result_path(out_dir, scenario_id, arm, run):
    return Path(out_dir) / f"{scenario_id}__{arm}__r{run}.json"


def _is_done(path, rerun_flags=RERUN_FLAGS):
    if not path.exists():
        return False
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False                                   # torn file from a killed session: redo
    return not any(prev.get("flags", {}).get(f) for f in rerun_flags)


def run_matrix(scenarios, arm_names, runs, make_llm, make_gate, base_clinic, out_dir,
               rerun_flags=RERUN_FLAGS, **orch_kw):
    """make_llm(arm) -> LLM client; make_gate(arm) -> gate or None. Returns counts of episodes run/skipped."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ran = skipped = 0
    for sc in scenarios:
        for arm_name in arm_names:
            for r in range(runs):
                path = result_path(out_dir, sc.id, arm_name, r)
                if _is_done(path, rerun_flags):
                    skipped += 1
                    continue
                arm = ARMS[arm_name]
                res = run_episode(sc, arm, make_llm(arm), make_gate(arm), base_clinic, **orch_kw)
                res["run"] = r
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
                os.replace(tmp, path)
                ran += 1
    return {"ran": ran, "skipped": skipped}
