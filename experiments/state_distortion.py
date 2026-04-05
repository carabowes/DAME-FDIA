import json
import numpy as np
import matplotlib.pyplot as plt


clean_path = "runs_live/ieee9/stealth/run_20260309_130707/clean.jsonl"
attack_path = "runs_live/ieee9/stealth/run_20260309_130707/attacked_estimates.jsonl"

bus_index = 3   # bus 4 → index 3


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


clean_rows = load_jsonl(clean_path)
attack_rows = load_jsonl(attack_path)


# map attacked estimates by timestep
attack_map = {r["t"]: r for r in attack_rows}


t = []
theta_true = []
theta_clean = []
theta_attack = []


for r in clean_rows:

    step = r["t"]

    if step not in attack_map:
        continue

    attack = attack_map[step]

    t.append(step)

    theta_true.append(r["x_true"][bus_index])
    theta_clean.append(r["x_hat"][bus_index])
    theta_attack.append(attack["x_hat_attacked"][bus_index])


t = np.array(t)
theta_true = np.array(theta_true)
theta_clean = np.array(theta_clean)
theta_attack = np.array(theta_attack)


plt.figure(figsize=(10,5))

plt.plot(t, theta_true, label="θ true", linewidth=2)
plt.plot(t, theta_clean, label="θ estimated (clean)", linewidth=2)
plt.plot(t, theta_attack, label="θ estimated (attacked)", linewidth=2)

plt.axvline(200, linestyle="--", color="red", label="attack start")
plt.axvline(260, linestyle="--", color="black", label="attack end")

plt.xlabel("Time step")
plt.ylabel("Bus angle θ")

plt.title("State Estimation Distortion under Stealth FDIA")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()