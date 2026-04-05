import json
import numpy as np
import matplotlib.pyplot as plt

# run_path = "runs_live/ieee9/stealth/run_20260306_164707"

#OCSVM STATE W5
# run_path = "runs_live/ieee9/stealth/run_20260308_172127"
# run_path = "runs_live/ieee9/stealth/run_20260309_000625"
run_path = "runs_live/ieee9/stealth/run_20260309_131209"
run_path = "runs_live/ieee9/stealth/run_20260309_205631"
#LOF STATE W 5
# run_path = "runs_live/ieee9/stealth/run_20260308_173155"

#IF W5
# run_path = "runs_live/ieee9/stealth/run_20260308_172731"

clean = {}
shift = []
t_vals = []

# Load clean estimates
with open(f"{run_path}/clean.jsonl") as f:
    for line in f:
        d = json.loads(line)
        clean[d["t"]] = np.array(d["x_hat"])

# Compare with attacked estimates
with open(f"{run_path}/attacked_estimates.jsonl") as f:
    for line in f:
        d = json.loads(line)

        t = d["t"]
        x_att = np.array(d["x_hat_attacked"])

        if t in clean:
            x_clean = clean[t]
            shift.append(np.linalg.norm(x_att - x_clean))
            t_vals.append(t)

shift = np.array(shift)
t_vals = np.array(t_vals)

plt.figure(figsize=(10,5))

plt.plot(t_vals, shift, linewidth=2, label="Estimator deviation")

plt.axvspan(200, 260, color="red", alpha=0.15, label="Attack window")

plt.title("State Estimation Shift Caused by Stealth FDIA")
plt.ylabel("State estimate deviation  ||x̂_attack − x̂_clean||")
plt.xlabel("Time step")

plt.legend()
plt.tight_layout()
plt.show()

print("Max estimator shift:", np.max(shift))
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# # run_path = "runs_live/ieee9/stealth/run_20260306_163258"
# run_path = "runs_live/ieee9/stealth/run_20260306_164707"
# # run_path = "runs_live/ieee9/standard/run_20260306_170905"

# clean = {}
# t_vals = []
# shift = []

# # load clean estimates
# with open(f"{run_path}/clean.jsonl") as f:
#     for line in f:
#         d = json.loads(line)
#         clean[d["t"]] = np.array(d["x_hat"])

# # compare with attacked estimates
# with open(f"{run_path}/attacked_estimates.jsonl") as f:
#     for line in f:
#         d = json.loads(line)

#         t = d["t"]
#         x_att = np.array(d["x_hat_attacked"])

#         if t in clean:
#             x_clean = clean[t]
#             shift.append(np.linalg.norm(x_att - x_clean))
#             t_vals.append(t)

# plt.figure(figsize=(10,5))
# plt.plot(t_vals, shift)

# plt.axvline(200, linestyle="--", color="red", label="Attack start")
# plt.axvline(260, linestyle="--", color="red", label="Attack end")

# plt.ylabel("State estimate deviation")
# plt.xlabel("Time step")
# plt.title("State estimation shift caused by stealth FDIA")
# plt.legend()
# plt.show()

# residual = []
# t = []

# with open(f"{run_path}/clean.jsonl") as f:
#     for line in f:
#         d = json.loads(line)
#         residual.append(d["residual_norm"])
#         t.append(d["t"])

# residual = np.array(residual)
# t = np.array(t)

# before = residual[t < 200]
# attack = residual[(t >= 200) & (t <= 260)]

# print("Residual statistics")
# print("Mean before:", np.mean(before))
# print("Mean attack:", np.mean(attack))
# print("Std before:", np.std(before))
# print("Std attack:", np.std(attack))

# bus_index = 3

# clean_vals = []
# att_vals = []
# t_vals = []

# for t in sorted(clean.keys()):
#     clean_vals.append(clean[t][bus_index])

# with open(f"{run_path}/attacked_estimates.jsonl") as f:
#     for line in f:
#         d = json.loads(line)

#         if d["t"] in clean:
#             att_vals.append(np.array(d["x_hat_attacked"])[bus_index])
#             t_vals.append(d["t"])

# plt.figure(figsize=(10,5))

# plt.plot(t_vals, clean_vals[:len(t_vals)], label="Clean estimate")
# plt.plot(t_vals, att_vals, label="Attacked estimate")

# # plt.axvline(200, linestyle="--", color="red")
# # plt.axvline(260, linestyle="--", color="red")
# plt.axvspan(200, 260, color="red", alpha=0.15, label="Attack window")

# plt.title("Bus angle estimate under stealth FDIA")
# plt.ylabel("Voltage angle (rad)")
# plt.xlabel("Time step")

# plt.legend()
# plt.show()

# # import json
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # run_path = "runs_live/ieee9/stealth/run_NEW"
# # # run_path = "runs_live/ieee9/stealth/run_20260306_161754"
# # # run_path = "runs_live/ieee9/stealth/run_20260306_160950"
# # run_path = "runs_live/ieee9/stealth/run_20260306_163258"
# # # run_path = "runs_live/ieee9/stealth/run_20260306_164707"

# # t = []
# # x_true = []
# # x_hat = []
# # residual = []
# # gen0 = []

# # with open(f"{run_path}/clean.jsonl") as f:
# #     for line in f:
# #         data = json.loads(line)

# #         t.append(data["t"])
# #         x_true.append(data["x_true"])
# #         x_hat.append(data["x_hat"])
# #         residual.append(data["residual_norm"])
# #         gen0.append(data["gen_p_mw"][0])

# # t = np.array(t)
# # x_true = np.array(x_true)
# # x_hat = np.array(x_hat)
# # residual = np.array(residual)

# # error = np.linalg.norm(x_true - x_hat, axis=1)

# # attack_start = 200
# # attack_end = 260

# # before = error[t < attack_start]
# # attack = error[(t >= attack_start) & (t <= attack_end)]
# # after = error[t > attack_end]

# # print("Mean estimation error")
# # print("Before attack:", np.mean(before))
# # print("During attack:", np.mean(attack))
# # print("After attack:", np.mean(after))

# # print("\nStd deviation")
# # print("Before attack:", np.std(before))
# # print("During attack:", np.std(attack))
# # print("After attack:", np.std(after))

# # rmse_before = np.sqrt(np.mean(before**2))
# # rmse_attack = np.sqrt(np.mean(attack**2))
# # rmse_after = np.sqrt(np.mean(after**2))

# # print("\nRMSE comparison")
# # print("Before:", rmse_before)
# # print("Attack:", rmse_attack)
# # print("After:", rmse_after)


# # # -------------------------
# # # State drift plot
# # # -------------------------

# # bus_index = 4   # corresponds to bus 4

# # theta_true = x_true[:, bus_index]
# # theta_hat = x_hat[:, bus_index]

# # plt.figure(figsize=(10,5))

# # plt.plot(t, theta_true, label="True angle")
# # plt.plot(t, theta_hat, label="Estimated angle")

# # plt.axvline(attack_start, linestyle="--", color="red")
# # plt.axvline(attack_end, linestyle="--", color="red")

# # plt.title("State estimation drift under stealth FDIA")
# # plt.xlabel("Time step")
# # plt.ylabel("Bus voltage angle (rad)")
# # plt.legend()

# # plt.show()


# # # -------------------------
# # # Attack magnitude plot
# # # -------------------------

# # attack_mag = []
# # attack_t = []

# # with open(f"{run_path}/attacked_estimates.jsonl") as f:
# #     for line in f:
# #         data = json.loads(line)

# #         attack_mag.append(data["attack_max_delta"])
# #         attack_t.append(data["t"])

# # attack_mag = np.array(attack_mag)
# # attack_t = np.array(attack_t)

# # plt.figure(figsize=(10,5))

# # plt.plot(attack_t, attack_mag, label="Attack magnitude")
# # plt.plot(t, error, label="Estimation error")

# # plt.axvline(attack_start, linestyle="--", color="red")
# # plt.axvline(attack_end, linestyle="--", color="red")

# # plt.title("Attack magnitude vs estimation error")
# # plt.xlabel("Time step")
# # plt.legend()

# # plt.show()


# # # -------------------------
# # # Residual statistics
# # # -------------------------

# # before_r = residual[t < attack_start]
# # attack_r = residual[(t >= attack_start) & (t <= attack_end)]

# # print("\nResidual statistics")
# # print("Mean before:", np.mean(before_r))
# # print("Mean attack:", np.mean(attack_r))

# # print("Std before:", np.std(before_r))
# # print("Std attack:", np.std(attack_r))

# # clean = {}
# # with open(f"{run_path}/clean.jsonl") as f:
# #     for line in f:
# #         d = json.loads(line)
# #         clean[d["t"]] = np.array(d["x_hat"])

# # shift = []
# # t_vals = []

# # with open(f"{run_path}/attacked_estimates.jsonl") as f:
# #     for line in f:
# #         d = json.loads(line)
# #         t = d["t"]
# #         x_att = np.array(d["x_hat_attacked"])

# #         if t in clean:
# #             x_clean = clean[t]
# #             shift.append(np.linalg.norm(x_att - x_clean))
# #             t_vals.append(t)

# # plt.plot(t_vals, shift)
# # plt.axvline(200, color="red", linestyle="--")
# # plt.axvline(260, color="red", linestyle="--")
# # plt.title("State estimation shift due to FDIA")
# # plt.show()