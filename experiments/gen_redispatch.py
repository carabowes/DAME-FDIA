import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---- RUN PATHS ----

run_no_mitigation = Path("runs_live/ieee9/stealth/run_20260327_132552")  # No mitigation baseline
run_with_mitigation = Path("runs_live/ieee9/stealth/run_20260327_131700")  # Mitigation + recovery


# ---- LOAD JSONL ----

def load_run(run_path):
    records = []
    with open(run_path / "attacked_estimates.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    
    t = []
    gen_p = []
    attack = []
    
    for r in records:
        if r.get("gen_p_pre") is None:
            continue
        
        t.append(r["t"])
        gen_p.append(r["gen_p_pre"][0])
        attack.append(r.get("attack_active", False))
    
    return np.array(t), np.array(gen_p), np.array(attack)


# LOAD RUNS

t_nm, gen_p_nm, attack_nm = load_run(run_no_mitigation)
t_wm, gen_p_wm, attack_wm = load_run(run_with_mitigation)


# COMPUTE GENERATOR POWER DEVIATIONS

baseline_nm = gen_p_nm[0]
dev_nm = gen_p_nm - baseline_nm

baseline_wm = gen_p_wm[0]
dev_wm = gen_p_wm - baseline_wm


# ATTACK WINDOW

attack_indices = np.where(attack_nm)[0]
attack_start = t_nm[attack_indices[0]] if len(attack_indices) > 0 else None
attack_end = t_nm[attack_indices[-1]] if len(attack_indices) > 0 else None


# PLOT

fig, ax = plt.subplots(figsize=(12, 5))

# Red line: No mitigation
ax.plot(
    t_nm,
    dev_nm,
    linewidth=2.5,
    label="No Mitigation",
    drawstyle='steps-pre',
    color='red',
    alpha=0.8
)

# Blue line: With mitigation + recovery
ax.plot(
    t_wm,
    dev_wm,
    linewidth=2.5,
    label="Mitigation + Recovery",
    drawstyle='steps-pre',
    color='green'
)

# Attack window shading
if attack_start is not None and attack_end is not None:
    ax.axvspan(
        attack_start,
        attack_end,
        alpha=0.15,
        color="red",
        label="Attack Active",
        zorder=0
    )

# Detection marker (alarm triggers)
if attack_start is not None:
    ax.axvline(
        attack_start,
        color='green',
        linestyle='--',
        linewidth=1.5,
        label='Detection (t=211)',
        zorder=2,
        alpha=0.7
    )

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Generator Power Deviation (MW)", fontsize=11)
ax.set_title("Closed-Loop Response: Generator Deviation (Detection → Mitigation → Recovery)", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle=':')
ax.legend(loc="center right", fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig('plots/figure_6_6_closed_loop_response.png', dpi=300, bbox_inches='tight')
print(" Saved: plots/figure_6_6_closed_loop_response.png")

plt.show()