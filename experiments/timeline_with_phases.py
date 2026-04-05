"""
Figure 6.6: Three-phase timeline showing detection → mitigation → recovery
Combines generator output, alarm trigger, measurement freezing, and recovery phases
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_jsonl(path):
    """Load JSONL records from path"""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_timeline_data(rows):
    """Extract all relevant timeline data"""
    t = []
    gen_output = []
    alarm = []
    frozen = []
    recovery = []
    
    for r in rows:
        if r.get("gen_p_pre") is None:
            continue
        
        t.append(r["t"])
        
        # Generator output (post-control)
        if r.get("gen_p_post") is not None:
            gen_output.append(r["gen_p_post"][0])
        else:
            gen_output.append(r["gen_p_pre"][0])
        
        alarm.append(1.0 if r["alarm"] else 0.0)
        frozen.append(1.0 if r["trusted_frozen"] else 0.0)
        recovery.append(1.0 if r["recovery_active"] else 0.0)
    
    return (
        np.array(t),
        np.array(gen_output),
        np.array(alarm),
        np.array(frozen),
        np.array(recovery),
    )


def main():
    # Load both runs
    run_no_recovery = Path("runs_live/ieee9/stealth/run_20260327_132552")  # no recovery
    run_with_recovery = Path("runs_live/ieee9/stealth/run_20260327_131700")  # with recovery
    
    rows_nr = load_jsonl(run_no_recovery / "attacked_estimates.jsonl")
    rows_wr = load_jsonl(run_with_recovery / "attacked_estimates.jsonl")
    
    t_nr, gen_nr, alarm_nr, frozen_nr, recovery_nr = extract_timeline_data(rows_nr)
    t_wr, gen_wr, alarm_wr, frozen_wr, recovery_wr = extract_timeline_data(rows_wr)
    
    # Baselines
    baseline_nr = gen_nr[0]
    gen_dev_nr = gen_nr - baseline_nr
    
    baseline_wr = gen_wr[0]
    gen_dev_wr = gen_wr - baseline_wr
    
    # ============================================================================
    # PLOT: Two-run comparison
    # ============================================================================
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # ---- TOP: No Recovery ----
    ax1.plot(t_nr, gen_dev_nr, 'b-', linewidth=2.5, label='Generator output deviation', zorder=3)
    ax1.axvspan(200, 260, alpha=0.12, color='red', label='Attack window', zorder=1)
    
    frozen_indices = np.where(frozen_nr)[0]
    if len(frozen_indices) > 0:
        frozen_starts = [t_nr[frozen_indices[0]]]
        frozen_ends = [t_nr[frozen_indices[0]]]
        for i in range(1, len(frozen_indices)):
            if frozen_indices[i] - frozen_indices[i-1] > 1:
                frozen_ends.append(t_nr[frozen_indices[i-1]])
                frozen_starts.append(t_nr[frozen_indices[i]])
        frozen_ends.append(t_nr[frozen_indices[-1]])
        for start, end in zip(frozen_starts, frozen_ends):
            ax1.axvspan(start, end, alpha=0.08, color='orange', zorder=0)
    
    alarm_indices = np.where(np.diff(alarm_nr, prepend=0) > 0)[0]
    if len(alarm_indices) > 0:
        alarm_start = t_nr[alarm_indices[0]]
        ax1.axvline(alarm_start, color='green', linestyle='--', linewidth=1.8, 
                   label='Alarm triggers', zorder=2)
    
    ax1.set_ylabel('Generator output deviation (MW)', fontsize=11)
    ax1.set_title('Detection & Mitigation Response (No Recovery)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    # ---- BOTTOM: With Recovery ----
    ax2.plot(t_wr, gen_dev_wr, 'b-', linewidth=2.5, label='Generator output deviation', zorder=3)
    ax2.axvspan(200, 260, alpha=0.12, color='red', label='Attack window', zorder=1)
    
    frozen_indices = np.where(frozen_wr)[0]
    if len(frozen_indices) > 0:
        frozen_starts = [t_wr[frozen_indices[0]]]
        frozen_ends = [t_wr[frozen_indices[0]]]
        for i in range(1, len(frozen_indices)):
            if frozen_indices[i] - frozen_indices[i-1] > 1:
                frozen_ends.append(t_wr[frozen_indices[i-1]])
                frozen_starts.append(t_wr[frozen_indices[i]])
        frozen_ends.append(t_wr[frozen_indices[-1]])
        for start, end in zip(frozen_starts, frozen_ends):
            ax2.axvspan(start, end, alpha=0.08, color='orange', zorder=0)
    
    alarm_indices = np.where(np.diff(alarm_wr, prepend=0) > 0)[0]
    if len(alarm_indices) > 0:
        alarm_start = t_wr[alarm_indices[0]]
        ax2.axvline(alarm_start, color='green', linestyle='--', linewidth=1.8, 
                   label='Alarm triggers', zorder=2)
    
    recovery_indices = np.where(recovery_wr)[0]
    if len(recovery_indices) > 0:
        recovery_start = t_wr[recovery_indices[0]]
        recovery_end = t_wr[recovery_indices[-1]]
        ax2.axvspan(recovery_start, recovery_end, alpha=0.1, color='green', 
                   label='Recovery phase', zorder=0)
    
    ax2.set_xlabel('Time step', fontsize=11)
    ax2.set_ylabel('Generator output deviation (MW)', fontsize=11)
    ax2.set_title('Detection, Mitigation & Recovery', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig('timeline_phases.png', dpi=300, bbox_inches='tight')
    print("Saved: timeline_phases.png")
    plt.show()


if __name__ == "__main__":
    main()
