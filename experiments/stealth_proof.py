import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Run path (no-defense run so attack effect is unmitigated)
# -----------------------------
run_path = Path("runs_live/ieee9/stealth/run_20260309_152738")

TARGET_BUS_STATE = 3  # index into x_hat (bus 4, zero-indexed after slack removal)

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
resid_clean = np.array([clean_map[ts]["residual_norm"] for ts in timesteps])
resid_att = np.array([est_map[ts]["residual_norm_attacked"] for ts in timesteps])

x_true = np.array([clean_map[ts]["x_true"][TARGET_BUS_STATE] for ts in timesteps])
x_hat_clean = np.array([clean_map[ts]["x_hat"][TARGET_BUS_STATE] for ts in timesteps])
x_hat_att = np.array([est_map[ts]["x_hat_attacked"][TARGET_BUS_STATE] for ts in timesteps])

attack_active = np.array([est_map[ts]["attack_active"] for ts in timesteps])

# Attack window
attack_idx = np.where(attack_active)[0]
if len(attack_idx) > 0:
    attack_start = t[attack_idx[0]]
    attack_end = t[attack_idx[-1]]
else:
    attack_start = attack_end = None

# -----------------------------
# Plot
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# --- Top panel: Residual norms ---
ax1.plot(t, resid_clean, linewidth=1.5, label="Clean residual", color="tab:blue")
ax1.plot(t, resid_att, linewidth=1.5, linestyle="--", label="Attacked residual", color="tab:red")

if attack_start is not None:
    ax1.axvspan(attack_start, attack_end, alpha=0.15, color="red", label="Attack window")

ax1.set_ylabel("Residual norm", fontsize=14)
ax1.set_title("Stealth FDIA: Residual Unchanged, State Corrupted", fontsize=16)
ax1.tick_params(labelsize=12)
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# --- Bottom panel: State estimate at target bus ---
ax2.plot(t, x_hat_clean, linewidth=1.5, label="Clean estimate", color="tab:blue")
ax2.plot(t, x_hat_att, linewidth=1.5, linestyle="--", label="Attacked estimate", color="tab:red")

if attack_start is not None:
    ax2.axvspan(attack_start, attack_end, alpha=0.15, color="red", label="Attack window")

ax2.set_xlabel("Time step", fontsize=14)
ax2.set_ylabel(f"Bus angle (Bus {TARGET_BUS_STATE + 1})", fontsize=14)
ax2.tick_params(labelsize=12)
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stealth_proof.png", dpi=300)
plt.show()

print(f"Attack window: t={attack_start} to t={attack_end}")
print(f"Mean residual (clean):    {resid_clean.mean():.4f}")
print(f"Mean residual (attacked): {resid_att.mean():.4f}")
