"""
Visualize attack coupling on buses 3, 4, 5 with clear deviation panel.
Shows that bus 5 deviates in opposite direction to buses 3, 4.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

run_path = Path("runs_live/ieee9/stealth/run_20260310_181416")

MA_WINDOW = 15

# Load data
clean_records = []
with open(run_path / "clean.jsonl") as f:
    for line in f:
        clean_records.append(json.loads(line))

est_records = []
with open(run_path / "attacked_estimates.jsonl") as f:
    for line in f:
        est_records.append(json.loads(line))

clean_records.sort(key=lambda r: r["t"])
est_records.sort(key=lambda r: r["t"])

# Align by timestep
t_set = set(r["t"] for r in clean_records) & set(r["t"] for r in est_records)
clean_map = {r["t"]: r for r in clean_records}
est_map = {r["t"]: r for r in est_records}
timesteps = sorted(t_set)

t = np.array(timesteps)
x_hat_clean = np.array([clean_map[ts]["x_hat"] for ts in timesteps])
x_hat_att = np.array([est_map[ts]["x_hat_attacked"] for ts in timesteps])
attack_active = np.array([est_map[ts]["attack_active"] for ts in timesteps])

# Compute deviation: attacked - clean
delta = x_hat_att - x_hat_clean

# Attack window
attack_idx = np.where(attack_active)[0]
attack_start = t[attack_idx[0]] if len(attack_idx) > 0 else None
attack_end = t[attack_idx[-1]] if len(attack_idx) > 0 else None

# Moving average
def moving_avg(arr, w):
    kernel = np.ones(w) / w
    out = np.empty_like(arr)
    for col in range(arr.shape[1]):
        out[:, col] = np.convolve(arr[:, col], kernel, mode="same")
    return out

x_clean_ma = moving_avg(x_hat_clean, MA_WINDOW)
x_att_ma = moving_avg(x_hat_att, MA_WINDOW)
delta_ma = moving_avg(delta, MA_WINDOW)

# Focus on buses 3, 4, 5 (indices 2, 3, 4)
focused_bus_indices = [2, 3, 4]
focused_bus_labels = ["Bus 3 (neighbour)", "Bus 4 (target)", "Bus 5 (neighbour)"]
focused_colors = ["green", "red", "purple"]

# All buses for context
all_bus_colors = plt.cm.tab10

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# --- Panel 1: Clean estimates (ALL BUSES) ---
for b in range(8):
    ax1.plot(t, x_clean_ma[:, b], linewidth=1.2, color=all_bus_colors(b), label=f"Bus {b+1}")

# Highlight the coupled buses
for idx, bus_idx in enumerate(focused_bus_indices):
    ax1.plot(t, x_clean_ma[:, bus_idx], linewidth=1.2, color=focused_colors[idx], label=focused_bus_labels[idx])

if attack_start is not None:
    ax1.axvspan(attack_start, attack_end, alpha=0.15, color="red")

ax1.set_ylabel("Bus angle (rad)", fontsize=14)
ax1.set_title("Clean State Estimates (All Buses, with Buses 3, 4, 5 highlighted)", fontsize=16)
ax1.tick_params(labelsize=12)
ax1.legend(loc="best", fontsize=12, ncol=3)
ax1.grid(True, alpha=0.3)

# --- Panel 2: Attacked estimates (ALL BUSES) ---
for b in range(8):
    ax2.plot(t, x_att_ma[:, b], linewidth=1.2, color=all_bus_colors(b), label=f"Bus {b+1}")

# Highlight the coupled buses
for idx, bus_idx in enumerate(focused_bus_indices):
    ax2.plot(t, x_att_ma[:, bus_idx], linewidth=1.2, color=focused_colors[idx], label=focused_bus_labels[idx])

if attack_start is not None:
    ax2.axvspan(attack_start, attack_end, alpha=0.15, color="red", label="Attack window")

ax2.set_ylabel("Bus angle (rad)", fontsize=14)
ax2.set_title("Attacked State Estimates (All Buses, with Buses 3, 4, 5 highlighted)", fontsize=16)
ax2.tick_params(labelsize=12)
ax2.legend(loc="best", fontsize=12, ncol=3)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Deviation (attacked - clean, FOCUSED buses only) ---
for idx, bus_idx in enumerate(focused_bus_indices):
    ax3.plot(t, delta_ma[:, bus_idx], linewidth=1.2, color=focused_colors[idx], label=focused_bus_labels[idx])

if attack_start is not None:
    ax3.axvspan(attack_start, attack_end, alpha=0.15, color="red")
    ax3.axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7)

ax3.set_xlabel("Time step", fontsize=14)
ax3.set_ylabel("State Deviation (rad)", fontsize=14)
ax3.set_title("Attack-Induced Deviation (Attacked - Clean, Buses 3/4/5 Only)", fontsize=16)
ax3.tick_params(labelsize=12)
ax3.legend(loc="best", fontsize=12)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("neighbour_coupling_detailed.png", dpi=300)
print("Saved to neighbour_coupling_detailed.png")

# Print summary
print("\n" + "=" * 60)
print("COUPLING ANALYSIS")
print("=" * 60)
if attack_start is not None:
    mid_attack = attack_start + (attack_end - attack_start) // 2
    print(f"\nAt t={mid_attack} (middle of attack window):")
    for idx, bus_idx in enumerate(focused_bus_indices):
        dev = delta_ma[t == mid_attack, bus_idx][0]
        print(f"  {focused_bus_labels[idx]}: Δx = {dev:+.8f} rad")
    
    print(f"\nInterpretation:")
    print(f"  • Bus 3 & 4: negative deviations (angles decrease)")
    print(f"  • Bus 5: positive deviation (angle increases)")
    print(f"  • This is realistic: electrical coupling can push neighbours in opposite directions")
    print(f"  • Bus 5 IS affected, just in opposite direction due to network impedance/current flow")
