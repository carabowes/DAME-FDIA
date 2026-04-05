import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --- EXACT RUNS YOU PROVIDED ---

runs = {
    "Residuals": "runs_live/ieee9/stealth/run_20260324_215411",
    "Prediction Error": "runs_live/ieee9/stealth/run_20260324_215429",
    "State": "runs_live/ieee9/stealth/run_20260324_215453"
}


# --- LOAD JSONL ---

def load_jsonl(path):

    rows = []

    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


# --- COMPUTE METRICS ---

def compute_metrics(run_dir):

    df = pd.DataFrame(
        load_jsonl(Path(run_dir) / "attacked_estimates.jsonl")
    )

    df["attack_active"] = df["attack_active"].astype(bool)
    df["alarm"] = df["alarm"].astype(bool)

    TP = ((df.alarm) & (df.attack_active)).sum()
    FP = ((df.alarm) & (~df.attack_active)).sum()
    FN = ((~df.alarm) & (df.attack_active)).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1


# --- COMPUTE F1 FOR EACH REPRESENTATION ---

labels = []
f1_scores = []

for rep, run_path in runs.items():

    precision, recall, f1 = compute_metrics(run_path)

    print(
        f"{rep}:  Precision={precision:.3f}  "
        f"Recall={recall:.3f}  F1={f1:.3f}"
    )

    labels.append(rep)
    f1_scores.append(f1)


# --- PLOT ---
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(6,5))

plt.bar(labels, f1_scores)

plt.ylabel("F1 Score")
plt.title("Stealth FDIA Detection by Feature Representation")

plt.ylim(0,1.05)

# plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()

plt.savefig("representation_comparison.png", dpi=300)

plt.show()