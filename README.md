# Q-Guard: Auditing and Repairing Argmax Collapse in Multi-Agent Quorum Systems

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange)](https://github.com/Xuefeng-Du1121/Q-Guard)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Partial%20Release-yellow)](https://github.com/Xuefeng-Du1121/Q-Guard)

> **⚠️ Partial Release Notice**  
> This repository contains the **inference-only code and pre-trained models** accompanying our paper submitted to IEEE Transactions on Mobile Computing. The complete codebase, including training scripts, hyperparameters, and experiment automation, will be released upon paper acceptance.

---

## Overview

**Q-Guard** is a deployment-time framework for auditing and repairing argmax collapse in multi-agent reinforcement learning (MARL) systems that use quorum-based coordination. We demonstrate that deterministic argmax execution breaks Byzantine-tolerant consensus in an Internet-of-Vehicles (IoV) system, dropping confirmation F1 from 0.673 (sampling) to 0.068, despite an unchanged policy.

### The Problem: Mode–Rate Mismatch

Multi-agent policies encode low-probability coordination actions as small per-decision hazards (~0.12). Sampling integrates these over time; argmax thresholds them away. Sub-modal agents rarely commit, the quorum loses its marginal members, and the system collapses.

### The Solution: Q-Guard

Q-Guard provides a three-stage workflow:

1. **Audit**: Compute quorum reach rate **Q** from protocol logs (no labels, no retraining)
2. **Diagnose**: Localize failures with commitment ratio **κ** and a quorum-formation funnel
3. **Repair**: Apply **Rate-Preserving Execution (RPE)**, a deterministic error-diffusion operator

**Key Results** (30 checkpoints, SUMO + ns-3 802.11p):
- Argmax collapses in 29/30 checkpoints (F1 < half of sampling)
- Q-Guard alarms on 28/30 (Q ratio < 0.5)
- RPE beats argmax on all 30 and passes Q-Guard on all 30
- Pooled gap closure: 90.2%

---

## What's Included

This partial release contains:

- ✅ Pre-trained actor weights (inference only)
- ✅ RPE operator implementation
- ✅ Q-Guard audit metrics (Q, κ, funnel)
- ✅ Public IoV-UAV evaluation simulator
- ✅ Inference API and examples
- ❌ Training code (released upon acceptance)
- ❌ Optimizer states and replay buffers
- ❌ Hyperparameters and training configurations
- ❌ Experiment automation scripts

---

## Installation

```bash
git clone https://github.com/Xuefeng-Du1121/Q-Guard.git
cd Q-Guard
pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.0+

---

## Quick Start

### Inference with RPE

```python
from qguard import QGuardPolicy, RPEOperator
from qguard.envs import IoVEnv

# Load pre-trained policy
policy = QGuardPolicy.from_pretrained("weights/checkpoint.pt")
rpe = RPEOperator()

# Run evaluation
env = IoVEnv(seed=42)
obs = env.reset()

for step in range(100):
    # Get policy distribution
    logits = policy.forward(obs)
    
    # Apply RPE (deterministic, rate-preserving)
    actions = rpe.execute(logits)
    
    obs, reward, done, info = env.step(actions)
    if done:
        break
```

### Q-Guard Audit

```python
from qguard.audit import compute_Q, compute_kappa

# From protocol logs (no labels needed)
Q = compute_Q(events_reported, events_reaching_committee)
kappa = compute_kappa(events_executed, event_hazard_sum)

# Check alarm
if Q < 0.5 * Q_baseline:
    print("⚠️ Q-Guard alarm: quorum collapse detected")
```

---

## Repository Structure

```
Q-Guard/
├── qguard/
│   ├── model.py           # Actor architecture (inference only)
│   ├── rpe.py             # Rate-Preserving Execution operator
│   ├── audit.py           # Q-Guard audit metrics
│   ├── envs/              # Public IoV-UAV simulator
│   └── consensus/         # SPBFT consensus protocol
├── weights/               # Pre-trained checkpoints
├── configs/               # Evaluation configurations
├── examples/              # Usage examples
└── tests/                 # Unit tests
```

---

## Citation

**Our paper is currently under review.** If you use this code, please cite:

```bibtex
@article{qguard2026,
  title={Q-Guard: Auditing and Repairing Argmax Collapse in Multi-Agent Quorum Systems},
  author={[Authors withheld for review]},
  journal={Under Review at IEEE Transactions on Mobile Computing},
  year={2026}
}
```

**Full citation with author names will be updated upon acceptance.**

---

## Roadmap

### Current Release (v0.1.0 - Partial)
- ✅ Inference code and pre-trained models
- ✅ RPE operator
- ✅ Q-Guard audit tools
- ✅ Public simulator

### Post-Acceptance Release (v1.0.0 - Complete)
- 🔜 Full training code
- 🔜 Experiment automation
- 🔜 Hyperparameters and training configurations
- 🔜 Ablation study scripts
- 🔜 Pre-registration records (SHA-256 verified)

---

## License

This code is released under the [MIT License](LICENSE).

**Usage Restrictions (Until Paper Acceptance):**
- ✅ You may use this code for research, education, and evaluation
- ✅ You may cite our work in your papers
- ❌ Please do not use this code as a baseline in submissions to conferences/journals before our paper is accepted
- ❌ Please do not re-train or modify the method for publication before the full code is released

These restrictions will be removed when the complete codebase is released upon paper acceptance.

---

## Contact

For questions about the code or paper, please open an issue in this repository.

**Note:** Author contact information will be added upon paper acceptance.

---

## Acknowledgments

We thank the reviewers and colleagues who provided feedback on this work. This research uses PyTorch, SUMO, and ns-3.

---

**Status:** 🟡 Partial Release (Inference Only) | 🔜 Full Release Upon Paper Acceptance
