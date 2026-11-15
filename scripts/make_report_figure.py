"""Figure 2 of the report: real-model agent results and the guard mutation test.

  python scripts/make_report_figure.py     (reads outputs/results/ablation_all.json and mutation.json, writes
                                            report/figures/agent_results.png)

(a) harmful attack episodes per arm over the pooled adversarial runs, with the exact 95% interval, and the harmful
    attempts split into executed and blocked;
(b) the guard mutation test: attack episodes (of 57) that still harm the guard-on arm when one rule, or two, is broken.
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/figures"
OUT.mkdir(parents=True, exist_ok=True)

abl = json.loads((ROOT / "outputs/results/ablation_all.json").read_text(encoding="utf-8"))
mut = json.loads((ROOT / "outputs/results/mutation.json").read_text(encoding="utf-8"))

arms = ["A0", "A1", "+Gate", "+Guard", "Full"]
n = [abl["arms"][a]["n"] for a in arms]
harm = [abl["arms"][a]["harm"] for a in arms]
lo = [abl["arms"][a]["harm_ci"][0] * 100 for a in arms]
hi = [abl["arms"][a]["harm_ci"][1] * 100 for a in arms]
rate = [h / k * 100 for h, k in zip(harm, n)]
attempts = [abl["arms"][a]["attempt"] for a in arms]
executed = harm                                   # an executed harmful attempt is a harmed episode
blocked = [t - e for t, e in zip(attempts, executed)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
xs = range(len(arms))
colors = ["#9aa5b1" if not a.endswith(("Guard", "Full")) else "#2f7d6d" for a in arms]
ax1.bar(xs, rate, color=colors, width=0.55)
ax1.errorbar(xs, rate, yerr=[[r - l for r, l in zip(rate, lo)], [h - r for r, h in zip(rate, hi)]], fmt="none",
             ecolor="black", capsize=3, lw=1)
for x, h, k, r in zip(xs, harm, n, rate):
    ax1.text(x, r + 0.35 if r else 0.35, f"{h}/{k}", ha="center", va="bottom", fontsize=8)
ax1.set_xticks(list(xs))
labels = [f"{a}\n{e + b} tried\n({e} ran)" if e + b else f"{a}\n0 tried"
          for a, e, b in zip(arms, executed, blocked)]
ax1.set_xticklabels(labels, fontsize=7.5)
ax1.set_ylabel("episodes with an unapproved action or leak (%)", fontsize=8)
ax1.set_title("(a) real models, 138 attack scenarios per arm", fontsize=9)
ax1.set_ylim(0, 6.5)
ax1.tick_params(axis="y", labelsize=8)

order = ["open_all_tools", "no_confirmation+no_ownership", "no_confirmation+no_patient_mismatch", "no_patient_mismatch",
         "no_confirmation+verify_any_phone", "no_confirmation", "no_ownership", "loose_affirmation", "confirm_any_action",
         "no_allowlist", "no_lock"]
names = {"open_all_tools": "identity gate removed", "no_confirmation+no_ownership": "no confirmation + no ownership",
         "no_confirmation+no_patient_mismatch": "no confirmation + no id match", "no_patient_mismatch": "patient-id match removed",
         "no_confirmation+verify_any_phone": "no confirmation + any phone", "no_confirmation": "confirmation removed",
         "no_ownership": "ownership removed", "loose_affirmation": "any message counts as yes",
         "confirm_any_action": "confirm covers any action", "no_allowlist": "allow-list removed", "no_lock": "lock removed"}
vals = [mut["mutants"][k]["attacks"]["harmed"] for k in order]
probes = [len(mut["mutants"][k]["attacks"]["probes_harmed"]) for k in order]
ys = list(range(len(order)))[::-1]
ax2.barh(ys, vals, color="#c0504d", height=0.6)
for y, v, p in zip(ys, vals, probes):
    ax2.text(v + 0.8, y, f"{v}/57" + (f"  (+{p} probe{'s' if p > 1 else ''})" if p else ""), va="center", fontsize=7)
ax2.set_yticks(ys)
ax2.set_yticklabels([names[k] for k in order], fontsize=7.5)
ax2.set_xlim(0, 80)
ax2.set_xlabel("attack episodes still harmful with the guard on", fontsize=8)
ax2.tick_params(axis="x", labelsize=8)
ax2.set_title("(b) one broken guard rule at a time (obedient agent)", fontsize=9)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "agent_results.png", dpi=200)
shutil.copy(ROOT / "outputs/figures/script_recall_test.png", OUT / "script_recall_test.png")
print("wrote", OUT / "agent_results.png")
