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
               rerun_flags=RERUN_FLAGS, progress=None, **orch_kw):
    """make_llm(arm) -> LLM client; make_gate(arm) -> gate or None. progress(result) is called after each episode
    is written. Returns counts of episodes run/skipped."""
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
                if progress is not None:
                    progress(res)
    return {"ran": ran, "skipped": skipped}


def quick_summary(out_dir):
    """Per-arm monitoring numbers over the result files in out_dir (not the formal analysis): episodes, share with
    task success, share with an unapproved action or a leak (judged against the scenario gold), mean tokens, and how
    many episodes carry a flag that makes them invalid for analysis."""
    arms, flagged = {}, {}
    for p in sorted(Path(out_dir).glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        a = arms.setdefault(r["arm"], {"episodes": 0, "task_success": 0, "unapproved": 0, "leak": 0, "tokens": 0,
                                       "escalated": 0, "auto_confirmations": 0})
        a["auto_confirmations"] += r.get("auto_confirmations", 0)
        s = r["score"]
        a["episodes"] += 1
        a["task_success"] += bool(s["task_success"])
        a["unapproved"] += s["unapproved_actions"] > 0
        a["leak"] += s["leaks"] > 0
        a["tokens"] += r.get("tokens", 0)
        a["escalated"] += r.get("escalation_turn") is not None
        for f, v in r.get("flags", {}).items():
            if v:
                flagged[f] = flagged.get(f, 0) + 1
    for a in arms.values():
        n = a["episodes"]
        a.update({"task_success": a["task_success"] / n, "unapproved": a["unapproved"] / n,
                  "leak": a["leak"] / n, "tokens": a["tokens"] / n, "escalated": a["escalated"] / n,
                  "auto_confirmations": a["auto_confirmations"] / n})
    return {"arms": arms, "flagged_episodes": flagged}
