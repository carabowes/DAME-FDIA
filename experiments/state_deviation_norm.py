import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_states(rows):
    t = []
    states = []

    for r in rows:
        if r.get("x_hat_used") is None:
            continue

        t.append(r["t"])
        states.append(r["x_hat_used"])

    return np.array(t), np.array(states)


def compute_deviation(states):

    baseline = states[0]

    dev = np.linalg.norm(states - baseline, axis=1)

    return dev


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--estimates", required=True)
    parser.add_argument("--out", default="state_deviation.png")

    args = parser.parse_args()

    rows = load_jsonl(args.estimates)

    t, states = extract_states(rows)

    deviation = compute_deviation(states)

    plt.figure(figsize=(10,5))

    plt.plot(t, deviation, linewidth=2)

    plt.xlabel("Time step")
    plt.ylabel("State deviation ||x - x0||")

    plt.title("System state deviation under FDIA")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(args.out, dpi=300)

    print("Saved plot:", args.out)


if __name__ == "__main__":
    main()