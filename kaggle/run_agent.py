"""Run the agent experiment (the arms of the 2x2) on Kaggle or locally. Resumable: finished episodes are skipped.

Kaggle (attach: the swasthya-data bundle, the G1 adapter dataset, optionally a scenarios dataset; Internet On; T4 GPU
only if a gate arm is run):
  from kaggle_secrets import UserSecretsClient; import os
  os.environ["LLM_API_KEY"] = UserSecretsClient().get_secret("LLM_API_KEY")      # never print it
  os.environ["LLM_API_BASE"] = "https://<your provider>/v1"                      # OpenAI-compatible endpoint
  !python $KD/run_agent.py --preflight                       # does the API tool-call in English and Hindi?
  !python $KD/run_agent.py --scenarios smoke --arms A1 +Guard --runs 1             # pilot, no GPU needed
  !python $KD/run_agent.py --scenarios smoke --arms +Gate Full --runs 1            # pilot with the tuned gate
  !python $KD/run_agent.py --scenarios /kaggle/input/<scenarios>/scenarios.jsonl --runs 3   # main run

Arms without a gate ("A0", "A1", "+Guard") need no GPU and no adapter. "+Gate" and "Full" load G1 from the adapter;
"+Gate_V0" loads the prompted Qwen gate. The gate probabilities are cached in <out>/gate_cache_*.jsonl, keyed by the
adapter's file hash. Results go to <out>/episodes/, one json per (scenario, arm, run). The API key is read from the
environment only and is never written to any file.
"""
import argparse
import glob
import json
import os
import sys
import time

for _p in ["src"] + glob.glob("/kaggle/input/**/src", recursive=True):
    if os.path.isdir(os.path.join(_p, "swasthya")):
        sys.path.insert(0, _p)
        ROOT = os.path.dirname(os.path.abspath(_p))
        break
else:
    raise SystemExit("cannot find src/swasthya (is the dataset attached?)")

from swasthya import kio, prompts  # noqa: E402
from swasthya.clinic import make_demo_clinic  # noqa: E402
from swasthya.gate import CachedGate, checkpoint_id_from_dir, prompted_checkpoint_id  # noqa: E402
from swasthya.llm import OpenAICompatClient  # noqa: E402
from swasthya.orchestrator import ARMS, scenario_gate_inputs  # noqa: E402
from swasthya.preflight import run_preflight  # noqa: E402
from swasthya.runner import quick_summary, run_matrix  # noqa: E402
from swasthya.scenario import check_scenarios, load_scenarios  # noqa: E402
from swasthya.smoke_scenarios import SMOKE_SCENARIOS  # noqa: E402

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_ARMS = ["A0", "A1", "+Gate", "+Guard", "Full"]          # proposal Table 1; "+Gate_V0" is opt-in


def make_client(a):
    key = os.environ.get(a.api_key_env)
    if not key:
        sys.exit(f"set the {a.api_key_env} environment variable (Kaggle: kaggle_secrets.UserSecretsClient)")
    base = a.base_url or os.environ.get("LLM_API_BASE")
    if not base:
        sys.exit("give --base_url or set LLM_API_BASE (an OpenAI-compatible endpoint, ending in /v1)")
    return OpenAICompatClient(a.model, base, key, temperature=0.0, max_tokens=a.max_tokens, retries=a.retries,
                              min_interval=a.min_interval), base


def find_adapter(a):
    if a.adapter_dir:
        return a.adapter_dir
    found = sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True))
    if len(found) != 1:
        sys.exit(f"found {len(found)} adapter folders under /kaggle/input {found}; pass --adapter_dir")
    return found[0]


def build_gates(arm_names, a):
    need = {ARMS[n].gate for n in arm_names} & {"g1", "v0"}
    gates = {}
    if "g1" in need:
        adapter = find_adapter(a)
        backend = None
        if not a.gate_cache_only:
            from swasthya.gate_backend import load_backend
            backend = load_backend(adapter_dir=adapter, bits=a.bits)
        gates["g1"] = CachedGate(backend, checkpoint_id_from_dir(adapter, "g1"),
                                 os.path.join(a.out, "gate_cache_g1.jsonl"), a.emergency_threshold)
    if "v0" in need:
        shots = prompts.pick_shots(kio.load_split(a.data_dir, "train"), 8, seed=13)
        backend = None
        if not a.gate_cache_only:
            from swasthya.gate_backend import load_backend
            backend = load_backend(adapter_dir=None, bits=a.bits, shots=shots, max_len=1536)
        gates["v0"] = CachedGate(backend, prompted_checkpoint_id("Qwen/Qwen2.5-3B-Instruct", shots),
                                 os.path.join(a.out, "gate_cache_v0.jsonl"), a.emergency_threshold)
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="smoke", help='"smoke" (5 built-in) or a .jsonl/.json scenario file')
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenarios")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default="/kaggle/working/agent" if os.path.isdir("/kaggle/working") else "outputs/agent")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--api_key_env", default="LLM_API_KEY")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--min_interval", type=float, default=1.0, help="seconds between API calls (free tiers rate-limit)")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--token_budget", type=int, default=20000, help="orchestrator tokens per episode")
    ap.add_argument("--gate_context", type=int, default=3, help="messages the gate reads (1 = latest patient message)")
    ap.add_argument("--adapter_dir", default=None, help="G1 adapter folder (default: the one under /kaggle/input)")
    ap.add_argument("--gate_cache_only", action="store_true", help="never load the gate model; cache misses fail closed")
    ap.add_argument("--emergency_threshold", type=float, default=None)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 16])
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data/d3_triage/rendered"))
    ap.add_argument("--preflight", action="store_true", help="check the API (tool calls, Hindi, repeatability) and exit")
    a = ap.parse_args()

    client, base = make_client(a)
    if a.preflight:
        res = run_preflight(client)
        for c in res["checks"]:
            print(f"{'ok  ' if c['ok'] else 'FAIL'} {'hard' if c['hard'] else 'soft'}  {c['name']:<28}"
                  f"{c['seconds']:>6}s  {c['detail']}")
        print("PREFLIGHT", "PASSED" if res["ok"] else "FAILED: do not start the run")
        sys.exit(0 if res["ok"] else 1)

    scenarios = SMOKE_SCENARIOS if a.scenarios == "smoke" else load_scenarios(a.scenarios)
    if a.limit:
        scenarios = scenarios[:a.limit]
    base_clinic = make_demo_clinic()
    problems = check_scenarios(scenarios, base_clinic)
    if problems:
        sys.exit("scenario gold problems, fix before running:\n" + json.dumps(problems, indent=1))
    os.makedirs(a.out, exist_ok=True)
    cfg = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "model": a.model, "base_url": base, "temperature": 0.0,
           "arms": a.arms, "runs": a.runs, "n_scenarios": len(scenarios), "scenarios": a.scenarios,
           "gate_context": a.gate_context, "token_budget": a.token_budget, "max_tokens": a.max_tokens,
           "emergency_threshold": a.emergency_threshold}
    gates = build_gates(a.arms, a)
    for arm in a.arms:                              # single-message gate input can be computed in one batch
        g = gates.get(ARMS[arm].gate)
        if g is not None and g.backend is not None and a.gate_context == 1:
            print(f"warming the {ARMS[arm].gate} gate cache: {g.warm(scenario_gate_inputs(scenarios, 1))} new texts")
    cfg["gate_checkpoints"] = {k: g.checkpoint_id for k, g in gates.items()}
    sessions_path = os.path.join(a.out, "run_config.json")
    sessions = json.load(open(sessions_path, encoding="utf-8")) if os.path.exists(sessions_path) else []
    sessions.append(cfg)
    json.dump(sessions, open(sessions_path, "w", encoding="utf-8"), indent=1)

    total, done, t0 = len(scenarios) * len(a.arms) * a.runs, [0], time.time()

    def progress(r):
        done[0] += 1
        s, f = r["score"], [k for k, v in r["flags"].items() if v]
        print(f"[{done[0]}] {r['scenario_id']:<22}{r['arm']:<9}r{r['run']}  success={s['task_success']!s:<5} "
              f"unapproved={s['unapproved_actions']} leaks={s['leaks']} tokens={r['tokens']}"
              f"{'  FLAGS ' + ','.join(f) if f else ''}", flush=True)

    ep_dir = os.path.join(a.out, "episodes")
    stats = run_matrix(scenarios, a.arms, a.runs, lambda arm: client, lambda arm: gates.get(arm.gate),
                       base_clinic, ep_dir, progress=progress, gate_context=a.gate_context,
                       token_budget=a.token_budget)
    summ = quick_summary(ep_dir)
    summ.update({"run": stats, "seconds": round(time.time() - t0), "episodes_planned": total})
    json.dump(summ, open(os.path.join(a.out, "run_summary.json"), "w", encoding="utf-8"), indent=1)
    print(f"\nran {stats['ran']}, skipped {stats['skipped']} (planned {total}) in {summ['seconds']}s")
    for arm, v in summ["arms"].items():
        print(f"  {arm:<9} n={v['episodes']:<4} success={v['task_success']:.2f} unapproved={v['unapproved']:.2f} "
              f"leak={v['leak']:.2f} escalated={v['escalated']:.2f} tokens/ep={v['tokens']:.0f}")
    if summ["flagged_episodes"]:
        print("FLAGGED episodes (invalid for analysis, rerun this command to retry):", summ["flagged_episodes"])


if __name__ == "__main__":
    main()
