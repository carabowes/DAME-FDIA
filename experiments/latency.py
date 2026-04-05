import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Choose run
# -----------------------------

# run_path = Path("runs_live/ieee9/stealth/run_20260307_172735")
run_path = Path("runs_live/ieee9/stealth/run_20260309_000625")
# run_path = Path("runs_live/ieee9/standard/run_20260307_174850")

#standard ocsvm
# run_path = Path("runs_live/ieee9/standard/run_20260307_133552")

#standard 4000 T
# run_path = Path("runs_live/ieee9/standard/run_20260308_125358")

#random
# run_path = Path("runs_live/ieee9/random/run_20260307_144602")
#random 4000
# run_path = Path("runs_live/ieee9/random/run_20260308_125545")

# -----------------------------
# Load records
# -----------------------------
print(f"\nAnalysing run: {run_path}\n")

# -----------------------------
# Load records
# -----------------------------

records = []
with open(run_path / "attacked_estimates.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

records = sorted(records, key=lambda r: r["t"])

t_vals = np.array([r["t"] for r in records])
attack_active = np.array([bool(r["attack_active"]) for r in records])
alarms = np.array([bool(r["alarm"]) for r in records])

# -----------------------------
# Find attack episodes
# -----------------------------

episodes = []
in_episode = False
start_idx = None

for i, active in enumerate(attack_active):

    if active and not in_episode:
        in_episode = True
        start_idx = i

    elif not active and in_episode:
        in_episode = False
        episodes.append((start_idx, i - 1))

if in_episode:
    episodes.append((start_idx, len(attack_active) - 1))

# -----------------------------
# Compute detection latencies
# -----------------------------

latencies = []
relative_latencies = []
missed = 0

for start_i, end_i in episodes:

    attack_start_t = t_vals[start_i]
    attack_end_t = t_vals[end_i]

    attack_duration = attack_end_t - attack_start_t

    alarm_indices = np.where(alarms[start_i:end_i + 1])[0]

    if len(alarm_indices) == 0:
        missed += 1
    else:

        first_alarm_i = start_i + alarm_indices[0]
        first_alarm_t = t_vals[first_alarm_i]

        latency = first_alarm_t - attack_start_t
        latencies.append(latency)

        if attack_duration > 0:
            relative_latencies.append(latency / attack_duration)

# -----------------------------
# Print results
# -----------------------------

print("Episodes found:", len(episodes))
print("Detected episodes:", len(latencies))
print("Missed episodes:", missed)

if len(episodes) > 0:
    detection_rate = len(latencies) / len(episodes)
else:
    detection_rate = 0

print(f"\nDetection rate: {detection_rate:.3f}")

if latencies:

    print("\nLatency statistics:")
    print("Mean latency:", np.mean(latencies))
    print("Median latency:", np.median(latencies))
    print("95th percentile latency:", np.percentile(latencies,95))

    if relative_latencies:
        print("\nRelative latency statistics:")
        print("Mean relative delay:", np.mean(relative_latencies))
        print("Median relative delay:", np.median(relative_latencies))

else:
    print("\nNo detected episodes.")

# -----------------------------
# Plot detection timeline
# -----------------------------

plt.figure(figsize=(10,4))

# plot alarms as events
alarm_times = t_vals[alarms]

plt.scatter(
    alarm_times,
    np.ones_like(alarm_times),
    color="black",
    s=20,
    label="Alarm"
)

for start_i, end_i in episodes:

    attack_start = t_vals[start_i]
    attack_end = t_vals[end_i]

    plt.axvspan(
        attack_start,
        attack_end,
        color="red",
        alpha=0.1,
        label="Attack window"
    )

# # remove duplicate legend labels
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))

# plt.xlabel("Time step")
# plt.ylabel("Alarm event")
# plt.title("Attack Detection Timeline")
# plt.ylim(0.8,1.2)

# plt.legend(by_label.values(), by_label.keys())

# plt.tight_layout()
# plt.show()

# -----------------------------
# # Plot latency distribution
# # -----------------------------

# if latencies:

#     plt.figure(figsize=(7,4))

#     bins = np.arange(-0.5, max(latencies)+1.5, 1)

#     plt.hist(
#         latencies,
#         bins=bins,
#         edgecolor="black",
#         alpha=0.8
#     )

#     plt.xticks(range(0, max(latencies)+1))

#     plt.xlabel("Detection latency (timesteps)")
#     plt.ylabel("Number of attacks")
#     plt.title("Detection Latency Distribution")

#     plt.tight_layout()
#     plt.show()