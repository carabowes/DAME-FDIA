#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


FEATURE_ALIASES = {
    "residuals": "residuals",
    "residual": "residuals",
    "state": "state",
    "prediction error": "prediction error",
    "prediction error": "prediction error",
}

PLOT_ORDER = ["residuals", "state", "prediction error"]
PLOT_LABELS = {
    "residuals": "Residuals",
    "state": "State",
    "prediction error": "Prediction Error",
}
PLOT_COLORS = {
    "residuals": "#1f77b4",
    "state": "#2ca02c",
    "prediction error": "#d62728",
}


def normalise_feature(x: str) -> str:
    if pd.isna(x):
        return ""
    return FEATURE_ALIASES.get(str(x).strip().lower(), str(x).strip().lower())


def load_jsonl(path: Path):
    """Load JSONL file and return list of rows."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_f1_from_run(run_dir: Path) -> float:
    """Compute F1 score from a run's attacked_estimates.jsonl."""
    rows = load_jsonl(run_dir / "attacked_estimates.jsonl")
    df = pd.DataFrame(rows)
    df["attack_active"] = df["attack_active"].astype(bool)
    df["alarm"] = df["alarm"].astype(bool)

    TP = ((df["alarm"]) & (df["attack_active"])).sum()
    FP = ((df["alarm"]) & (~df["attack_active"])).sum()
    FN = ((~df["alarm"]) & (df["attack_active"])).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def df_from_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load manifest CSV and compute F1 for each run."""
    man = pd.read_csv(manifest_path)
    required = {"run_dir", "feature", "window_size"}
    missing = required - set(man.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    out_rows = []
    for idx, r in man.iterrows():
        run_dir = Path(str(r["run_dir"]))
        if not run_dir.exists():
            print(f"  [WARN] Run dir not found: {run_dir}, skipping")
            continue
        try:
            f1 = compute_f1_from_run(run_dir)
            row_dict = {
                "feature": r["feature"],
                "window_size": r["window_size"],
                "f1": f1,
                "run_dir": str(run_dir),
            }
            if "seed" in man.columns:
                row_dict["seed"] = r["seed"]
            if "scenario" in man.columns:
                row_dict["scenario"] = r["scenario"]
            if "detector" in man.columns:
                row_dict["detector"] = r["detector"]
            out_rows.append(row_dict)
        except Exception as e:
            print(f"  [ERROR] Failed to process {run_dir}: {e}")
            continue
    
    if not out_rows:
        raise ValueError("No runs successfully processed from manifest")
    
    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Window Size vs F1 (Residual/State/Prediction Error)"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Precomputed CSV with columns: feature, window_size, f1",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Run manifest CSV with columns: run_dir, feature, window_size, [seed, scenario, detector]",
    )
    parser.add_argument(
        "--manifests",
        default=None,
        help="Comma-separated list of manifest files for multi-scenario subfigures",
    )
    parser.add_argument(
        "--out",
        default="window_size_vs_f1.png",
        help="Output image path",
    )
    parser.add_argument(
        "--study",
        default=None,
        help="Optional filter, e.g. 'Stealth Attacks'",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional filter, e.g. stealth, random, standard",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated list of scenarios for subfigures (e.g. 'stealth,standard,random')",
    )
    parser.add_argument(
        "--detector",
        default="ocsvm",
        help="Optional filter for detector (default: ocsvm). Use 'all' to disable.",
    )
    parser.add_argument(
        "--agg",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation across seeds/runs per (feature, window_size)",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Create separate PNG files for each scenario instead of subfigures",
    )

    args = parser.parse_args()

    # Determine if multi-scenario subfigure mode
    use_subfigures = args.scenarios is not None and not args.separate
    
    if use_subfigures:
        # Multi-scenario mode: create subfigures
        scenario_list = [s.strip() for s in args.scenarios.split(",")]
        manifest_paths = args.manifests.split(",") if args.manifests else None
        
        if manifest_paths and len(manifest_paths) != len(scenario_list):
            raise ValueError("--manifests must have same count as --scenarios")
        
        # Load all data from single or multiple manifests
        all_dfs = []
        if args.manifest:
            # Single unified manifest with scenario column
            df_all = df_from_manifest(Path(args.manifest))
        else:
            # Multiple manifests, one per scenario
            df_list = []
            for i, scen in enumerate(scenario_list):
                if not manifest_paths:
                    raise ValueError("Need --manifests for multi-scenario mode")
                man_path = manifest_paths[i].strip()
                print(f"Loading {man_path} for scenario '{scen}'...")
                df_temp = df_from_manifest(Path(man_path))
                if "scenario" not in df_temp.columns:
                    df_temp["scenario"] = scen
                df_list.append(df_temp)
            df_all = pd.concat(df_list, ignore_index=True)
        
        plot_scenarios(df_all, scenario_list, args, use_subfigures=True)
    elif args.separate and args.scenarios:
        # Separate plots mode: create individual files for each scenario
        scenario_list = [s.strip() for s in args.scenarios.split(",")]
        manifest_paths = args.manifests.split(",") if args.manifests else None
        
        if manifest_paths and len(manifest_paths) != len(scenario_list):
            raise ValueError("--manifests must have same count as --scenarios")
        
        # Load and plot each scenario separately
        for i, scen in enumerate(scenario_list):
            if not manifest_paths:
                raise ValueError("Need --manifests for separate mode")
            man_path = manifest_paths[i].strip()
            print(f"\nLoading {man_path} for scenario '{scen}'...")
            df_temp = df_from_manifest(Path(man_path))
            if "scenario" not in df_temp.columns:
                df_temp["scenario"] = scen
            
            # Generate output filename
            out_base = Path(args.out)
            out_name = out_base.stem + f"_{scen}" + out_base.suffix
            out_path = out_base.parent / out_name
            
            # Plot individual scenario
            plot_single_scenario(df_temp, scen, args, out_path)
    else:
        # Single-scenario mode: original behavior
        # Load data from either manifest or CSV
        if args.manifest:
            df = df_from_manifest(Path(args.manifest))
        elif args.csv:
            df = pd.read_csv(args.csv)
        else:
            raise ValueError("Provide either --csv, --manifest, or --manifests+--scenarios")

        plot_scenarios(df, [args.scenario] if args.scenario else None, args, use_subfigures=False)


def plot_scenarios(df, scenario_list, args, use_subfigures=False) -> None:

    required_cols = {"feature", "window_size", "f1"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Make column access robust to capitalisation differences.
    col_map = {c.lower(): c for c in df.columns}
    df = df.rename(columns={col_map["feature"]: "feature",
                            col_map["window_size"]: "window_size",
                            col_map["f1"]: "f1"})

    if "study" in df.columns and args.study:
        df = df[df["study"].astype(str).str.lower() == args.study.lower()]

    if "scenario" in df.columns and args.scenario:
        df = df[df["scenario"].astype(str).str.lower() == args.scenario.lower()]

    if "detector" in df.columns and args.detector and args.detector.lower() != "all":
        df = df[df["detector"].astype(str).str.lower() == args.detector.lower()]

    df["feature"] = df["feature"].map(normalise_feature)
    df["window_size"] = pd.to_numeric(df["window_size"], errors="coerce")
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df = df.dropna(subset=["feature", "window_size", "f1"])

    df = df[df["feature"].isin(PLOT_ORDER)]

    if df.empty:
        raise ValueError("No rows left after filtering. Check --study/--scenario/--detector.")

    # Setup style and figure
    plt.style.use("seaborn-v0_8-whitegrid")
    
    LINE_STYLES = {
        "residuals": "-",
        "state": "-",
        "prediction error": "--",
    }

    if use_subfigures:
        # Multi-scenario mode: create 1x3 subfigures
        n_scenarios = len(scenario_list)
        fig, axes = plt.subplots(1, n_scenarios, figsize=(5.5 * n_scenarios, 5), sharey=True)
        if n_scenarios == 1:
            axes = [axes]
        
        for ax_idx, scenario in enumerate(scenario_list):
            ax = axes[ax_idx]
            
            # Filter data for this scenario
            if "scenario" in df.columns:
                df_scen = df[df["scenario"].astype(str).str.lower() == scenario.lower()]
            else:
                df_scen = df
            
            if df_scen.empty:
                print(f"  [WARN] No data for scenario: {scenario}")
                continue
            
            # Group by feature and window_size
            grouped_list = []
            for (feat, ws), sub in df_scen.groupby(["feature", "window_size"]):
                f1_vals = sub["f1"].values
                grouped_list.append({
                    "feature": feat,
                    "window_size": ws,
                    "f1_mean": f1_vals.mean(),
                    "f1_std": f1_vals.std() if len(f1_vals) > 1 else 0.0,
                    "n_seeds": len(f1_vals),
                })
            grouped = pd.DataFrame(grouped_list).sort_values(["feature", "window_size"])
            
            # Plot each feature
            for feature in PLOT_ORDER:
                sub = grouped[grouped["feature"] == feature].sort_values("window_size")
                if sub.empty:
                    continue
                
                x = sub["window_size"].values
                y = sub["f1_mean"].values
                y_std = sub["f1_std"].values
                
                # Plot line with markers
                ax.plot(x, y, linewidth=2.5, linestyle=LINE_STYLES[feature],
                       label=PLOT_LABELS[feature], color=PLOT_COLORS[feature],
                       marker='o', markersize=7)
                
                # Add error band
                ax.fill_between(x, y - y_std, y + y_std,
                               alpha=0.15, color=PLOT_COLORS[feature])
            
            # Format subplot
            ax.set_xlabel("Window Size", fontsize=18)
            if ax_idx == 0:
                ax.set_ylabel("F1-score", fontsize=18)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            
            # Format scenario title (capitalize)
            title = scenario.capitalize() if scenario else "Unknown"
            ax.set_title(f"({chr(97 + ax_idx)}) {title} FDIA", fontsize=18)
            ax.tick_params(labelsize=16)
            
            # Print summary for this scenario
            print(f"\n{title} FDIA:")
            for feature in PLOT_ORDER:
                sub = grouped[grouped["feature"] == feature].sort_values("window_size")
                if sub.empty:
                    continue
                print(f"\n  {PLOT_LABELS[feature]}:")
                for _, row in sub.iterrows():
                    print(f"    Window {int(row['window_size'])}: {row['f1_mean']:.4f} ± {row['f1_std']:.4f} ({int(row['n_seeds'])} seeds)")
        
        # Shared legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), 
                      ncol=3, fontsize=18, frameon=True)
        
        plt.tight_layout()
    else:
        # Single-scenario mode: original single-plot behavior
        fig = plt.figure(figsize=(9, 6))
        
        # Group by feature and window_size
        grouped_list = []
        for (feat, ws), sub in df.groupby(["feature", "window_size"]):
            f1_vals = sub["f1"].values
            grouped_list.append({
                "feature": feat,
                "window_size": ws,
                "f1_mean": f1_vals.mean(),
                "f1_std": f1_vals.std() if len(f1_vals) > 1 else 0.0,
                "n_seeds": len(f1_vals),
            })
        grouped = pd.DataFrame(grouped_list).sort_values(["feature", "window_size"])
        
        # Plot each feature
        for feature in PLOT_ORDER:
            sub = grouped[grouped["feature"] == feature].sort_values("window_size")
            if sub.empty:
                continue
            
            x = sub["window_size"].values
            y = sub["f1_mean"].values
            y_std = sub["f1_std"].values
            
            # Plot line with markers
            plt.plot(x, y, linewidth=2.5, linestyle=LINE_STYLES[feature],
                    label=PLOT_LABELS[feature], color=PLOT_COLORS[feature],
                    marker='o', markersize=7)
            
            # Add error band
            plt.fill_between(x, y - y_std, y + y_std,
                           alpha=0.15, color=PLOT_COLORS[feature])
        
        plt.xlabel("Window Size", fontsize=18)
        plt.ylabel("F1-score", fontsize=18)
        title = "Detection Performance vs Window Size"
        if scenario_list and scenario_list[0]:
            title += f" ({scenario_list[0].capitalize()} Attacks)"
        plt.title(title, fontsize=22)
        plt.tick_params(labelsize=16)
        plt.ylim(0, 1.05)
        plt.legend(loc="upper right", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        print("\nAggregated points used for plot (mean ± std):")
        for feature in PLOT_ORDER:
            sub = grouped[grouped["feature"] == feature].sort_values("window_size")
            if sub.empty:
                continue
            print(f"\n{PLOT_LABELS[feature]}:")
            for _, row in sub.iterrows():
                print(f"  Window {int(row['window_size'])}: {row['f1_mean']:.4f} ± {row['f1_std']:.4f} ({int(row['n_seeds'])} seeds)")
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {out_path}")


def plot_single_scenario(df, scenario, args, out_path) -> None:
    """Plot a single scenario as a standalone figure."""
    required_cols = {"feature", "window_size", "f1"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Make column access robust to capitalisation
    col_map = {c.lower(): c for c in df.columns}
    df = df.rename(columns={col_map["feature"]: "feature",
                            col_map["window_size"]: "window_size",
                            col_map["f1"]: "f1"})

    if "detector" in df.columns and args.detector and args.detector.lower() != "all":
        df = df[df["detector"].astype(str).str.lower() == args.detector.lower()]

    df["feature"] = df["feature"].map(normalise_feature)
    df["window_size"] = pd.to_numeric(df["window_size"], errors="coerce")
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df = df.dropna(subset=["feature", "window_size", "f1"])

    df = df[df["feature"].isin(PLOT_ORDER)]

    if df.empty:
        raise ValueError(f"No data for scenario {scenario}")

    # Setup style and figure
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(9, 6))
    
    LINE_STYLES = {
        "residuals": "-",
        "state": "-",
        "prediction error": "-",
    }

    # Group by feature and window_size
    grouped_list = []
    for (feat, ws), sub in df.groupby(["feature", "window_size"]):
        f1_vals = sub["f1"].values
        grouped_list.append({
            "feature": feat,
            "window_size": ws,
            "f1_mean": f1_vals.mean(),
            "f1_std": f1_vals.std() if len(f1_vals) > 1 else 0.0,
            "n_seeds": len(f1_vals),
        })
    grouped = pd.DataFrame(grouped_list).sort_values(["feature", "window_size"])
    
    # Plot each feature
    for feature in PLOT_ORDER:
        sub = grouped[grouped["feature"] == feature].sort_values("window_size")
        if sub.empty:
            continue
        
        x = sub["window_size"].values
        y = sub["f1_mean"].values
        y_std = sub["f1_std"].values
        
        # Plot line with markers
        plt.plot(x, y, linewidth=2.5, linestyle=LINE_STYLES[feature],
                label=PLOT_LABELS[feature], color=PLOT_COLORS[feature],
                marker='o', markersize=7)
        
        # Add error band
        plt.fill_between(x, y - y_std, y + y_std,
                       alpha=0.15, color=PLOT_COLORS[feature])
    
    plt.xlabel("Window Size", fontsize=18)
    plt.ylabel("F1-score", fontsize=18)
    title_label = scenario.capitalize() if scenario else "Unknown"
    plt.title(f"Detection Performance vs Window Size ({title_label} FDIA)", fontsize=22)
    plt.tick_params(labelsize=16)
    plt.ylim(0, 1.05)
    plt.legend(loc="upper right", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Print summary
    print(f"\n{title_label} FDIA:")
    for feature in PLOT_ORDER:
        sub = grouped[grouped["feature"] == feature].sort_values("window_size")
        if sub.empty:
            continue
        print(f"\n  {PLOT_LABELS[feature]}:")
        for _, row in sub.iterrows():
            print(f"    Window {int(row['window_size'])}: {row['f1_mean']:.4f} ± {row['f1_std']:.4f} ({int(row['n_seeds'])} seeds)")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()