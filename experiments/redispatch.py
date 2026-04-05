import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# ---- RUN PATHS ----

# NO_RECOVERY_RUN = Path("runs_live/ieee9/stealth/run_20260308_142712") # stays permanently displaced.
# RECOVERY_RUN = Path("runs_live/ieee9/stealth/run_20260308_141355") #
# NO_DEFENSE_RUN = Path("runs_live/ieee9/stealth/run_20260308_141334")
# NO_RECOVERY_RUN = Path("runs_live/ieee9/stealth/run_20260327_114600") # mitigation + control (state detector)
# RECOVERY_RUN = Path("runs_live/ieee9/stealth/run_20260327_124443") # same run, shows direct restoration

#OCSVM STATE W5
# NO_RECOVERY_RUN = Path("runs_live/ieee9/stealth/run_20260308_234713") #control applied but no redispatch
NO_RECOVERY_RUN = Path("runs_live/ieee9/stealth/run_20260309_000417") #no control applied but no redispatch 
RECOVERY_RUN = Path("runs_live/ieee9/stealth/run_20260309_000625")
# NO_RECOVERY_RUN = Path("runs_live/ieee9/stealth/run_20260325_025529") #no control applied but no redispatch 
# RECOVERY_RUN = Path("runs_live/ieee9/stealth/run_20260325_025227")

# # NO_RECOVERY_RUN = Path("runs_live/ieee9/standard/run_20260401_184727") # mitigation + control only (no recovery)
# # RECOVERY_RUN = Path("runs_live/ieee9/standard/run_20260401_184352") # full closed-loop with recovery

# ---- LOAD JSONL ----

def load_jsonl(path):

    rows = []

    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


# ---- EXTRACT GENERATOR DATA ----

def extract_generator_data(rows):
    t = []
    gen_pre = []
    gen_post = []

    for r in rows:

        if r.get("gen_p_pre") is None:
            continue

        t.append(r["t"])
        gen_pre.append(r["gen_p_pre"])

        if r.get("gen_p_post") is None:
            gen_post.append(r["gen_p_pre"])
        else:
            gen_post.append(r["gen_p_post"])

    return np.array(t), np.array(gen_pre), np.array(gen_post)

def extract_alarm_data(rows):
    """Extract alarm timeline"""
    t = []
    alarm = []
    
    for r in rows:
        if r.get("t") is None:
            continue
        t.append(r["t"])
        alarm.append(1.0 if r.get("alarm", False) else 0.0)
    
    return np.array(t), np.array(alarm)


def main():
# ---- LOAD RUNS ----
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--out", default="recovery_comparison.png")

    args = parser.parse_args()

    rows_nr = load_jsonl(NO_RECOVERY_RUN / "attacked_estimates.jsonl")
    rows_r = load_jsonl(RECOVERY_RUN / "attacked_estimates.jsonl")

    t_nr, gen_pre_nr, gen_post_nr = extract_generator_data(rows_nr)
    t_r, gen_pre_r, gen_post_r = extract_generator_data(rows_r)
    
    # Extract alarm data from recovery run (detection moment only)
    t_alarm, alarm_signal = extract_alarm_data(rows_r)


    # ---- COMPUTE DEVIATION FROM BASELINE ----

    baseline = gen_pre_nr[0,0]   # generator 1 baseline from first run

    dev_nr = gen_post_nr[:,0] - baseline
    dev_r  = gen_post_r[:,0] - baseline
    
    # No smoothing - use raw data to show discrete-time controller behavior
    # The step response is accurate to the control model
    dev_r_smooth = dev_r
    
    # Find detection moment
    alarm_indices = np.where(np.diff(alarm_signal.astype(int), prepend=0) > 0)[0]
    detection_time = t_alarm[alarm_indices[0]] if len(alarm_indices) > 0 else None
    
    # ---- PLOT: Single clean plot with annotations ----
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    # Plot generator deviations
    ax.plot(t_nr, dev_nr, linewidth=2.5, label="Without Mitigation", 
            color='#d62728', alpha=0.85, zorder=3)
    ax.plot(t_r, dev_r_smooth, linewidth=2.5, label="With Mitigation & Recovery", 
            color='#1f77b4', zorder=3)
    
    # Attack window
    ax.axvspan(200, 260, color="red", alpha=0.1, zorder=0)
    
    # Recovery phase (after attack, before return to nominal)
    ax.axvspan(260, 320, color="green", alpha=0.05, zorder=0)
    
    # Detection trigger moment
    if detection_time is not None:
        ax.axvline(detection_time, color='green', linestyle='--', linewidth=2,
                   zorder=2, alpha=0.8)
    
    # Phase annotations
    # ax.text(230, 0.04, "ATTACK\nWINDOW", fontsize=9, fontweight='bold', 
    #         ha='center', color='#8B0000', alpha=0.7,
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # ax.text(detection_time, -0.85, "DETECTION\nTRIGGER", fontsize=8, fontweight='bold',
    #         ha='center', color='darkgreen', alpha=0.7,
    #         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.text(290, -0.25, "RECOVERY\nPHASE", fontsize=9, fontweight='bold',
            ha='center', color='darkgreen', alpha=0.6,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6, edgecolor='none'))
    
    # Main labels
    # ax.text(50, -0.78, "NORMAL\nOPERATION", fontsize=8.5, ha='center', 
    #         color='gray', alpha=0.6, style='italic')
    # ax.text(520, -0.78, "NOMINAL\nRESTORED", fontsize=8.5, ha='center',
    #         color='darkgreen', alpha=0.6, style='italic')
    
    # Legend with custom elements
    legend_elements = [
        plt.Line2D([0], [0], color='#d62728', linewidth=2.5, alpha=0.85, label='Without Mitigation'),
        plt.Line2D([0], [0], color='#1f77b4', linewidth=2.5, label='With Mitigation & Recovery'),
        # plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.1, label='Attack window'),
        plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.05, label='Recovery phase'),
        plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, alpha=0.8, label='First Detection'),
    ]
    ax.legend(handles=legend_elements, loc='center right', fontsize=9, framealpha=0.95)
    
    ax.set_xlabel("Time step", fontsize=11, fontweight='bold')
    ax.set_ylabel("Generator output deviation (MW)", fontsize=11, fontweight='bold')
    ax.set_title("Closed-Loop Mitigation and Recovery Response to FDIA", 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add explanatory text at bottom
    fig.text(0.5, 0.02, 
             "The controller applies corrective redispatch at the detection timestep, resulting in an immediate setpoint change in this discrete-time model. In practice, physical ramp rate constraints (±5 MW/step) would produce gradual convergence.",
             ha='center', fontsize=9, style='italic', color='#333333')
    
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim([-0.92, 0.08])
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print("Saved:", args.out)
    plt.show()

if __name__ == "__main__":
    main()    
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse


# def load_jsonl(path):
#     rows = []
#     with open(path) as f:
#         for line in f:
#             rows.append(json.loads(line))
#     return rows


# def extract_generator_data(rows):

#     t = []
#     gen_pre = []
#     gen_post = []
    
#     for r in rows:

#         if r.get("gen_p_pre") is None:
#             continue

#         t.append(r["t"])
#         gen_pre.append(r["gen_p_pre"])

#         if r.get("gen_p_post") is None:
#             gen_post.append(r["gen_p_pre"])
#         else:
#             gen_post.append(r["gen_p_post"])

#     return np.array(t), np.array(gen_pre), np.array(gen_post)


# def main():

#     parser = argparse.ArgumentParser()

#     parser.add_argument("--estimates", required=True)
#     parser.add_argument("--out", default="generator_redispatch.png")

#     args = parser.parse_args()

#     rows = load_jsonl(args.estimates)

#     t, gen_pre, gen_post = extract_generator_data(rows)
#     baseline = gen_pre[0]

#     dev_gen1 = gen_post[:,0] - baseline[0]
#     dev_gen2 = gen_post[:,1] - baseline[1]

#     plt.figure(figsize=(10,5))

#     # plt.plot(t, gen_pre[:,0], "--", label="Gen1 pre-control")
#     # plt.plot(t, gen_post[:,0], "-", label="Gen1 post-control")

#     # plt.plot(t, gen_pre[:,1], "--", label="Gen2 pre-control")
#     # plt.plot(t, gen_post[:,1], "-", label="Gen2 post-control")

#     plt.plot(t, dev_gen1, linewidth=2, label="Gen1 redispatch")

#     plt.plot(t, dev_gen2, linewidth=2, label="Gen2 redispatch")
    
    
#     plt.xlabel("Time step")
#     # plt.ylabel("Generator output (MW)")
#     plt.ylabel("Generator deviation from baseline (MW)")
#     plt.title("Generator redispatch under FDIA mitigation")

#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     plt.savefig(args.out, dpi=300)

#     print("Saved:", args.out)


# if __name__ == "__main__":
#     main()