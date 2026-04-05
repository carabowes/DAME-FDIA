import json
import numpy as np
import matplotlib.pyplot as plt

#defense but no recovery
# run_path = "runs_live/ieee9/stealth/run_20260306_140622"

#recovery here
run_path = "runs_live/ieee9/stealth/run_20260309_000625"

time = []
error = []
residual = []

with open(f"{run_path}/clean.jsonl") as f:

    for line in f:
        data = json.loads(line)

        t = data["t"]
        x_true = np.array(data["x_true"])
        x_hat = np.array(data["x_hat"])

        err = np.linalg.norm(x_true - x_hat)

        time.append(t)
        error.append(err)
        residual.append(data["residual_norm"])

# fig, ax1 = plt.subplots()

# ax1.plot(time, error, label="Estimation Error", color="blue")
# ax1.set_ylabel("||x_true - x_hat||")

# ax2 = ax1.twinx()
# ax2.plot(time, residual, label="Residual Norm", color="orange")
# ax2.set_ylabel("Residual")

# # plt.axvline(200, linestyle="--", color="red")
# # plt.axvline(260, linestyle="--", color="red")
# plt.axvline(200, linestyle="--", color="red", label="Attack start")
# plt.axvline(260, linestyle="--", color="red", label="Attack end")
# ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")
# plt.title("Stealth FDIA: Estimation Error vs Residual")
# plt.show()

import json
import numpy as np
import matplotlib.pyplot as plt

# run_path = "runs_live/ieee9/stealth/run_20260306_140622"
run_path = "runs_live/ieee9/stealth/run_20260309_000625"

time = []
error = []
gen0 = []
score_t = []
score_v = []
with open(f"{run_path}/clean.jsonl") as f:

    for line in f:
        data = json.loads(line)

        x_true = np.array(data["x_true"])
        x_hat = np.array(data["x_hat"])

        error.append(np.linalg.norm(x_true - x_hat))
        gen0.append(data["gen_p_mw"][0])
        time.append(data["t"])

with open(f"{run_path}/attacked_estimates.jsonl") as f:
    for line in f:
        data = json.loads(line)

        score_t.append(data["t"])
        score_v.append(data["score"])

# with open(f"{run_path}/attacked_estimates.jsonl") as f:

#     for line in f:
#         data = json.loads(line)
#         # score.append(data["score"])
#         score.append((data["t"], data["score"]))


fig, axs = plt.subplots(3,1, figsize=(10,8))

axs[0].plot(time, error)
axs[0].set_title("State Estimation Error")
axs[0].set_ylabel("||x_true - x_hat||")
# axs[1].plot(time, score)
axs[1].plot(score_t, score_v)
axs[1].set_ylabel("Detector Score")
axs[1].set_title("Detector Score")

axs[2].plot(time, gen0)
axs[2].set_ylabel("Generator MW")
axs[2].set_xlabel("Time step")
axs[2].set_title("Generator Output (Control Response)")

for ax in axs:
    
    ax.axvline(200, linestyle="--", color="red")
    ax.axvline(260, linestyle="--", color="red")

plt.tight_layout()
plt.show()
# import json
# import matplotlib.pyplot as plt

# run_path = "runs_live/ieee9/stealth/run_20260306_140622"
#  run_path = "runs_live/ieee9/stealth/run_20260306_145207"

# time = []
# theta_true = []
# theta_hat = []
# gen0 = []
# gen1 = []

# bus_index = 3   # θ3 for example


# with open(f"{run_path}/clean.jsonl") as f:

#     for line in f:

#         data = json.loads(line)

#         time.append(data["t"])

#         theta_true.append(data["x_true"][bus_index])
#         theta_hat.append(data["x_hat"][bus_index])

#         gen0.append(data["gen_p_mw"][0])
#         gen1.append(data["gen_p_mw"][1])


# fig, axes = plt.subplots(3,1, figsize=(10,10))


# # Panel A — estimator corruption
# axes[0].plot(time, theta_true, label="θ_true")
# axes[0].plot(time, theta_hat, label="θ_hat")
# axes[0].set_title("State estimation under FDIA")
# axes[0].legend()


# # Panel B — physical state
# axes[1].plot(time, theta_true, label="θ_true")
# axes[1].set_title("Physical bus angle response")
# axes[1].legend()


# # Panel C — generator redispatch
# axes[2].plot(time, gen0, label="Generator 0")
# axes[2].plot(time, gen1, label="Generator 1")
# axes[2].set_title("Generator redispatch")
# axes[2].set_xlabel("Time")
# axes[2].legend()


# for ax in axes:
#     ax.axvline(200, linestyle="--", color="red")
#     ax.axvline(260, linestyle="--", color="red")


# plt.tight_layout()
# plt.show()