# Reproducibility Guide

This guide describes the full end-to-end workflow used to generate dissertation results.

---

## End-to-End Workflow

### Step 1: Collect Clean Training Data

Generate feature data from a clean run (no attack applied).

```bash
PYTHONPATH=. python -m scripts.run_live_ieee9 \
  --scenario standard \
  --attack_schedule fixed \
  --attack_start 999999 \
  --attack_end 999999 \
  --detector_type none \
  --representation state \
  --window_size 5 \
  --log_features \
  --stop_after_steps 10000
```

**Output:** `runs_live/ieee9/standard/run_<timestamp>/features.jsonl`

Attack is disabled by setting start/end outside simulation horizon.

---

### Step 2: Train ML Detector

Train anomaly detection models using collected clean features.

```bash
PYTHONPATH=. python -m scripts.train_streaming_detectors \
  runs_live/ieee9/standard/run_<timestamp> \
  --window_size 5 \
  --representation state \
  --out_dir trained_detectors
```

**Output:** `trained_detectors/`
- `ocsvm_state_W5.pkl`
- `lof_state_W5.pkl`
- `iforest_state_W5.pkl`
- `scaler_state_W5.pkl`

---

### Step 3: Run Detection on Attack Scenarios

The framework supports three attack types: **random**, **standard**, and **stealth**.

---

#### Random Attack

Unstructured noise injected into measurements (detectable by all feature types):

```bash
python -m scripts.run_live_ieee9 \
  --scenario random \
  --attack_schedule random \
  --detector_type ocsvm \
  --representation residuals \
  --window_size 5 \
  --detector_dir trained_detectors \
  --stop_after_steps 1000
```

Expected: High detection performance across all feature representations.


#### Stanadrd Attack

Structured attack with fixed offsets applied to selected measurements:

```bash
python -m scripts.run_live_ieee9 \
  --scenario standard \
  --attack_schedule random \
  --detector_type ocsvm \
  --representation residuals \
  --window_size 5 \
  --detector_dir trained_detectors \
  --stop_after_steps 1000
```

Expected: Detectable using both residual-based and state-based features.

**Stealth Attack** (state features required):

```bash
python -m scripts.run_live_ieee9 \
  --scenario stealth \
  --attack_schedule fixed \
  --attack_start 200 \
  --attack_end 260 \
  --attack_strength 0.15 \
  --attack_bus 4 \
  --detector_type ocsvm \
  --representation state \
  --window_size 5 \
  --detector_dir trained_detectors \
  --stop_after_steps 600
```

**Output:** `runs_live/ieee9/stealth/run_<timestamp>/`

Expected F1: **~0.82** (state-based) vs 0.14 (residual-based)

---

### Step 4: Enable Mitigation

Freeze measurements when an anomaly is detected:

```bash
python -m scripts.run_live_ieee9 \
  --scenario stealth \
  --attack_schedule fixed \
  --attack_start 200 \
  --attack_end 260 \
  --attack_strength 0.15 \
  --attack_bus 4 \
  --detector_type ocsvm \
  --representation state \
  --window_size 5 \
  --detector_dir trained_detectors \
  --enable_mitigation \
  --stop_after_steps 600
```

Frozen measurements prevent state estimate corruption.

---

### Step 5: Enable Control and Recovery

Activate full closed-loop response with generator redispatch:

```bash
python -m scripts.run_live_ieee9 \
  --scenario stealth \
  --attack_schedule fixed \
  --attack_start 200 \
  --attack_end 260 \
  --attack_strength 0.15 \
  --attack_bus 4 \
  --detector_type ocsvm \
  --representation state \
  --window_size 5 \
  --detector_dir trained_detectors \
  --enable_mitigation \
  --enable_control \
  --control_on_alarm \
  --enable_recovery \
  --stop_after_steps 600
```

**Control Sequence:**
- On alarm: OPF solves for optimal generator redispatch (±10 MW/step)
- After alarm clears: Gradual ramp-down to baseline
- Recovery triggered after 10-step clean streak

**Key Factor:** The effectiveness of mitigation is influenced by detection latency, which determines how long corrupted estimates affect system behaviour before corrective action is applied.

---

### Step 6: Evaluate Results

Generate detection and mitigation metrics:

```bash
python scripts/evaluate_streaming_run.py runs_live/ieee9/stealth/run_<timestamp>
```

**Outputs:**
- Detection metrics (Precision, Recall, F1)
- State corruption profile
- Control actions
- Recovery behavior

Additional analysis:

```bash
python experiments/stealth_proof.py
python experiments/all_bus_angles.py
python experiments/window_comparison.py
```

---

## Key Parameters

| Parameter | Default | Note |
|-----------|---------|------|
| `--attack_start` | 200 | Attack onset step |
| `--attack_end` | 260 | Attack offset step |
| `--attack_strength` | 0.15 | 15% of state norm |
| `--attack_bus` | 4 | Target bus (localized) |
| `--window_size` | 5 | Temporal context (steps) |
| `--representation` | state | Feature type (state/residuals/innovations) |
| `--detector_type` | ocsvm | ML model (ocsvm/lof/isolation_forest) |

---

## Important Notes

- Clean data is required for training detectors
- State-based features required for stealth attack detection (residuals fail)
- Attack window duration: 60 steps (200–260)
- Detection threshold: calibrated from clean-period scores

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

67 test cases validate:
- Attack generation
- State estimation
- Feature extraction
- Control & recovery logic
- Mitigation metrics
