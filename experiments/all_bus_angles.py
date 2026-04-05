"""
All-bus state estimate comparison: normal operation vs stealth attack.
Shows moving-averaged bus angles for clean and attacked estimates across all buses.
Two panels: top = clean estimates, bottom = attacked estimates.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
# run_path = Path("runs_live/ieee9/stealth/run_20260309_152738")
# run_path = Path("runs_live/ieee9/stealth/run_20260310_175904") # 3 buses (3, 4 , 5)
# run_path = Path("runs_live/ieee9/stealth/run_20260310_180149") #just bus 3

run_path = Path("runs_live/ieee9/stealth/run_20260310_181416")

MA_WINDOW = 15  # moving average window size

# Bus labels: state index 0 = bus 1, ..., state index 7 = bus 8 (slack bus 0 removed)
BUS_LABELS = [f"Bus {i+1}" for i in range(8)]

# -----------------------------
# Load data
# -----------------------------
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
n_buses = len(clean_records[0]["x_hat"])

# (T, n_buses) arrays
x_hat_clean = np.array([clean_map[ts]["x_hat"] for ts in timesteps])
x_hat_att = np.array([est_map[ts]["x_hat_attacked"] for ts in timesteps])
attack_active = np.array([est_map[ts]["attack_active"] for ts in timesteps])

# Compute deviation: attacked - clean
delta = x_hat_att - x_hat_clean

# Attack window
attack_idx = np.where(attack_active)[0]
attack_start = t[attack_idx[0]] if len(attack_idx) > 0 else None
attack_end = t[attack_idx[-1]] if len(attack_idx) > 0 else None


# -----------------------------
# Moving average
# -----------------------------
def moving_avg(arr, w):
    """Moving average along axis 0 (time)."""
    kernel = np.ones(w) / w
    out = np.empty_like(arr)
    for col in range(arr.shape[1]):
        out[:, col] = np.convolve(arr[:, col], kernel, mode="same")
    return out


x_clean_ma = moving_avg(x_hat_clean, MA_WINDOW)
x_att_ma = moving_avg(x_hat_att, MA_WINDOW)
delta_ma = moving_avg(delta, MA_WINDOW)

# -----------------------------
# Plot
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

cmap = plt.cm.tab10

# --- Top: Clean estimates (all buses) ---
for b in range(n_buses):
    ax1.plot(t, x_clean_ma[:, b], linewidth=1.2, color=cmap(b), label=BUS_LABELS[b])

if attack_start is not None:
    ax1.axvspan(attack_start, attack_end, alpha=0.15, color="red", label="Attack window")

ax1.set_ylabel("Bus angle (rad)")
ax2.set_xlim(t[7], t[-7]) 
ax1.set_title("Clean State Estimates (Moving Average Window = 15)")
ax1.legend(loc="upper right", fontsize=7, ncol=4)
ax1.grid(True, alpha=0.3)

# --- Bottom: Attacked estimates (all buses) ---
for b in range(n_buses):
    ax2.plot(t, x_att_ma[:, b], linewidth=1.2, color=cmap(b), label=BUS_LABELS[b])

if attack_start is not None:
    ax2.axvspan(attack_start, attack_end, alpha=0.15, color="red", label="Attack window")

ax2.set_xlabel("Time step")
ax2.set_xlim(t[7], t[-7])
ax2.set_ylabel("Bus angle (rad)")
ax2.set_title("Attacked State Estimates (Moving Average Window = 15)")
ax2.legend(loc="upper right", fontsize=7, ncol=4)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("all_bus_angles_comparison2.png", dpi=300)
plt.show()

print(f"Buses plotted: {n_buses}")
print(f"Moving average window: {MA_WINDOW}")
print(f"Attack window: t={attack_start} to t={attack_end}")
