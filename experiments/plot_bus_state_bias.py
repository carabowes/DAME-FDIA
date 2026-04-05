import json
import numpy as np
import matplotlib.pyplot as plt

runs = {
    # 4: "runs_live/ieee9/stealth/run_20260306_221336",
    # 5: "runs_live/ieee9/stealth/run_20260306_233313",
    # 1: "runs_live/ieee9/stealth/run_20260306_233357",
    # 3: "runs_live/ieee9/stealth/run_20260306_233449",

    # state rep w5
    4: "runs_live/ieee9/stealth/run_20260309_000625",
    5: "runs_live/ieee9/stealth/run_20260309_003955",
    1: "runs_live/ieee9/stealth/run_20260309_004019",
    3: "runs_live/ieee9/stealth/run_20260309_004039",
}


attack_start = 200
attack_end = 260

plt.figure(figsize=(11,6))

for attacked_bus, run_path in runs.items():

    clean = {}
    attack = {}
    t_vals = []

    with open(f"{run_path}/clean.jsonl") as f:
        for line in f:
            d = json.loads(line)
            t = d["t"]
            clean[t] = np.array(d["x_hat"])
            t_vals.append(t)

    with open(f"{run_path}/attacked_estimates.jsonl") as f:
        for line in f:
            d = json.loads(line)
            attack[d["t"]] = np.array(d["x_hat_attacked"])

    t_vals = sorted(t_vals)

    delta = []

    for t in t_vals:

        if t in attack and t in clean:
            theta_clean = clean[t][attacked_bus]
            theta_attack = attack[t][attacked_bus]
            delta.append(theta_attack - theta_clean)
        else:
            delta.append(0)

    # plt.plot(t_vals, delta, label=f"Attack on Bus {attacked_bus}")
    plt.plot(t_vals, delta, linewidth=2, label=f"Attack on Bus {attacked_bus}")
# plt.axhline(0, linestyle="--")
plt.axvspan(attack_start, attack_end, color="red", alpha=0.12)

plt.xlabel("Time step")
plt.ylabel("Δθ (attack − clean) [rad]")
plt.title("Propagation of Stealth FDIA Bias Across IEEE-9 Buses")

plt.legend()
plt.tight_layout()
plt.show()
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# #bus 4
# run_path = "runs_live/ieee9/stealth/run_20260306_221336"

# #bus 5
# run_path = "runs_live/ieee9/stealth/run_20260306_233313"

# #bus 1
# run_path = "runs_live/ieee9/stealth/run_20260306_233357"

# #bus 3
# run_path = "runs_live/ieee9/stealth/run_20260306_233449"

# attack_start = 200
# attack_end = 260

# # buses to visualise
# buses = [1,3,4,5]

# clean = {}
# t_vals = []

# # load clean estimates
# with open(f"{run_path}/clean.jsonl") as f:
#     for line in f:
#         d = json.loads(line)
#         t = d["t"]

#         clean[t] = np.array(d["x_hat"])

#         if t not in t_vals:
#             t_vals.append(t)

# # load attacked estimates
# attack = {}

# with open(f"{run_path}/attacked_estimates.jsonl") as f:
#     for line in f:
#         d = json.loads(line)
#         attack[d["t"]] = np.array(d["x_hat_attacked"])

# t_vals = np.array(sorted(t_vals))

# plt.figure(figsize=(11,6))

# for bus in buses:

#     delta = []

#     for t in t_vals:

#         if t in attack and t in clean:

#             theta_clean = clean[t][bus]
#             theta_attack = attack[t][bus]

#             delta.append(theta_attack - theta_clean)

#         else:
#             delta.append(0)

#     delta = np.array(delta)

#     plt.plot(t_vals, delta, label=f"Bus {bus}")

# plt.axhline(0, linestyle="--", linewidth=1)

# plt.axvspan(attack_start, attack_end, color="red", alpha=0.15)

# plt.xlabel("Time step")
# plt.ylabel("Δθ (attack − clean) [rad]")
# plt.title("Propagation of Stealth FDIA Bias Across IEEE-9 Buses")

# plt.legend()

# plt.tight_layout()
# plt.show()
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# run_path = "runs_live/ieee9/stealth/run_20260306_221336"

# attack_start = 200
# attack_end = 260

# bus_index = 3  # change to test other buses

# t_vals = []
# delta_theta = []

# # load clean estimates
# clean = {}
# with open(f"{run_path}/clean.jsonl") as f:
#     for line in f:
#         d = json.loads(line)
#         clean[d["t"]] = d["x_hat"][bus_index]

# # load attacked estimates
# with open(f"{run_path}/attacked_estimates.jsonl") as f:
#     for line in f:
#         d = json.loads(line)

#         t = d["t"]

#         if t in clean:
#             theta_clean = clean[t]
#             theta_attack = d["x_hat_attacked"][bus_index]

#             delta = theta_attack - theta_clean

#             t_vals.append(t)
#             delta_theta.append(delta)

# t_vals = np.array(t_vals)
# delta_theta = np.array(delta_theta)

# plt.figure(figsize=(10,5))

# plt.plot(t_vals, delta_theta, linewidth=2, label="Attack-induced bias")

# plt.axhline(0, linestyle="--")

# plt.axvspan(attack_start, attack_end, color="red", alpha=0.15)

# plt.xlabel("Time step")
# plt.ylabel("Δθ (attack − clean) [rad]")
# plt.title("Attack-Induced Bias in Bus Voltage Angle Estimate")
# plt.legend()

# plt.tight_layout()
# plt.show()

# print("Max attack bias:", np.max(np.abs(delta_theta)))