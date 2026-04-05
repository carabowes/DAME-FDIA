import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# run_path = Path("runs_live/ieee9/stealth/run_20260307_172735")
# run_path = Path("runs_live/ieee9/stealth/run_20260309_000625")
# run_path = Path("runs_live/ieee9/stealth/run_20260309_141525")
run_path = Path("runs_live/ieee9/stealth/run_20260324_215453")

records = []
with open(run_path / "attacked_estimates.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

records = sorted(records, key=lambda r: r["t"])

t = np.array([r["t"] for r in records])
score = np.array([r["score"] for r in records])
attack = np.array([r["attack_active"] for r in records])

attack_start = t[np.where(attack)[0][0]]
attack_end = t[np.where(attack)[0][-1]]

# find first alarm
alarms = np.array([r["alarm"] for r in records])
alarm_indices = np.where((alarms == True) & (attack == True))[0]

first_alarm = None
if len(alarm_indices) > 0:
    first_alarm = t[alarm_indices[0]]
plt.figure(figsize=(10,4))

# anomaly score
plt.plot(t, score, linewidth=2, label="Anomaly score")

# attack window
plt.axvline(attack_start, color="red", linestyle="--")
plt.axvline(attack_end, color="red", linestyle="--")
plt.axvspan(200,260, alpha=0.15, color="red", label="Attack window")
# plot ALL alarm detections
alarm_indices = np.where(alarms == True)[0]

plt.scatter(t[alarm_indices],
            score[alarm_indices],
            color="red",
            s=30,
            label="Detections")

plt.xlabel("Time step", fontsize=14)
plt.ylabel("Anomaly score", fontsize=14)
plt.title("Anomaly Score over Time for Stealth FDIA Detection", fontsize=16)
plt.tick_params(labelsize=12)
plt.legend(loc="upper right", bbox_to_anchor=(1,1), fontsize=12)
plt.tight_layout()
plt.show()