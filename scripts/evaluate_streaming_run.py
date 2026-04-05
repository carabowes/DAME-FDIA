import json
import argparse
from pathlib import Path
import pandas as pd

def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def evaluate(run_dir: Path):

    est = load_jsonl(run_dir / "attacked_estimates.jsonl")
    df = pd.DataFrame(est)

    df["attack_active"] = df["attack_active"].astype(bool)
    df["alarm"] = df["alarm"].astype(bool)

    TP = ((df.alarm) & (df.attack_active)).sum()
    FP = ((df.alarm) & (~df.attack_active)).sum()
    FN = ((~df.alarm) & (df.attack_active)).sum()
    TN = ((~df.alarm) & (~df.attack_active)).sum()

    total_timesteps = len(df)
    total_attacked = df["attack_active"].sum()
    total_clean = total_timesteps - total_attacked
    total_alarms = df["alarm"].sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN)

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "total_timesteps": int(total_timesteps),
        "attack_timesteps": int(total_attacked),
        "clean_timesteps": int(total_clean),
        "total_alarms": int(total_alarms),
        "TP": int(TP),
        "FN": int(FN),
        "FP": int(FP),
        "TN": int(TN),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "accuracy": accuracy
    }

    return metrics

def main(run_dir: Path, save: bool):

    metrics = evaluate(run_dir)

    print("\n=== Streaming Detection Evaluation ===")

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:.4f}")
        else:
            print(f"{k:20}: {v}")

    if save:
        out_file = run_dir / "evaluation_metrics.json"
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nSaved metrics → {out_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    main(args.run_dir, args.save)