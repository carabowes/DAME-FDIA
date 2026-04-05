# FDIA Detection and Mitigation

This project presents a cyber-physical framework for detecting and mitigating False Data Injection Attacks (FDIA) in power system state estimation.

The framework operates in a **continuous, streaming setting**, integrating:

- state estimation  
- anomaly detection  
- mitigation, control, and recovery

within a closed-loop system.

**Key Idea:**
- Stealth FDIA attacks are constructed to preserve residuals
- They induce a shift in the estimated system state
- Detecting anomalies in the state estimate enables detection where residual-based methods fail

**Key Result:**
- State-based detection achieves ~0.82 F1 on stealth attacks  
- Residual-based detection fails

**Key Insight:**
- Detection alone is insufficient to prevent system impact  
- Mitigation and control must be integrated within the estimation–control loop  
- Detection latency directly influences system behaviour and recovery

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python -m scripts.run_live_ieee9 \
  --scenario stealth \
  --attack_start 200 \
  --attack_end 260 \
  --attack_strength 0.15 \
  --attack_bus 4 \
  --detector_type ocsvm \
  --representation state \
  --window_size 5 \
  --detector_dir trained_detectors_streaming_final \
  --enable_mitigation \
  --enable_control \
  --control_on_alarm \
  --enable_recovery \
  --stop_after_steps 600
```

## Project Structure

```
src/                        Core implementation of the framework
scripts/                    Entry points for running simulations and evaluations
tests/                      Unit, Integration, Reproduciblity tests
trained_detectors           Final trained models
runs_live/                  Simulation outputs
experiments/                Scripts used to generate evaluation results and figures
plots/                      Dissertation figures
```

## Reproducibility

For full end-to-end workflow (data generation → training → evaluation), see:

**[MANUAL.md](MANUAL.md)**