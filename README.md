# Hyperparameter-Invariant Spiking Neural Networks

This repository contains the implementation and experimental code for studying hyperparameter invariance in Spiking Neural Networks (SNNs).

## Research Overview

This work investigates the sensitivity of SNN performance to hyperparameter choices (particularly the membrane time constant τ) and proposes methods to achieve robustness.

### Key Findings

1. **Traditional LIF neurons** show significant accuracy variation (~2%) when τ changes
2. **AccumulatorLIF** achieves stable performance across different τ values (std < 0.3%)
3. **Layer normalization** further improves robustness

## Experimental Results

### CIFAR-10 Classification

| Method | Accuracy | τ Sensitivity (Std) |
|--------|----------|---------------------|
| Baseline Spikformer | 95.29% | ~2.1% |
| Ours (AccLIF) | 93.73% ± 0.16% | 0.3% |

### τ Ablation Study

| τ value | Baseline | Ours |
|---------|----------|------|
| 1.0 | 90.2% | 93.5% |
| 1.5 | 92.8% | 93.6% |
| 2.0 | 95.3% | 93.7% |
| 2.5 | 93.1% | 93.8% |
| 3.0 | 91.5% | 93.6% |

## Repository Structure

```
├── Core Implementation
│   ├── accumulator_lif.py          # AccumulatorLIF neuron (reset-free formulation)
│   ├── associative_scan.py         # Parallel scan algorithm (early exploration)
│   ├── deer_lif_node.py            # DEER-compatible LIF node
│   └── deer_model.py               # Model wrapper
│
├── Training Code
│   └── cifar10/
│       ├── train.py                # Main training script
│       ├── model.py                # Spikformer architecture
│       ├── model_deer.py           # Modified model with AccLIF
│       └── cifar10.yml             # Configuration
│
├── Experiment Scripts
│   ├── train_deer_94_config.py     # Main experiment
│   ├── train_deer_seeds.py         # Multi-seed experiments
│   ├── run_ablation_tau.py         # τ ablation
│   ├── run_ablation_T.py           # Time step ablation
│   └── test_hyperparameter_invariance.py
│
├── Results (JSON format)
│   ├── output_ablation_tau/        # τ ablation results
│   ├── output_ablation_T/          # T ablation results
│   ├── output_deer_94config_seeds/ # Multi-seed statistics
│   └── hyperparameter_invariance_results/
│
└── Figures
    └── paper_figures_final/        # Generated figures
```

## Environment Setup

```bash
conda create -n spike python=3.9
conda activate spike

pip install torch==1.10.0+cu111 torchvision
pip install spikingjelly==0.0.0.0.12
pip install timm==0.5.4
pip install cupy-cuda11x
pip install pyyaml matplotlib numpy
```

## Running Experiments

### Train Model
```bash
python train_deer_94_config.py --seed 42 --epochs 300
```

### Multi-seed Experiments
```bash
python train_deer_seeds.py
```

### τ Ablation Study
```bash
python run_ablation_tau.py
```

### Test Hyperparameter Invariance
```bash
python test_hyperparameter_invariance.py
```

## Result Format

All results are stored in JSON format for reproducibility:

**results.json**:
```json
{
  "best_acc": 93.73,
  "best_epoch": 285,
  "total_time_hours": 14.13,
  "history": {
    "train_loss": [...],
    "train_acc": [...],
    "test_acc": [...]
  }
}
```

**config.json**: Complete experiment configuration

**summary_statistics.json**: Aggregated statistics across seeds

## Key Implementation Details

| File | Description |
|------|-------------|
| `accumulator_lif.py` | Reset-free LIF with adaptive threshold |
| `associative_scan.py` | Parallel prefix sum (exploratory work) |
| `train_deer_94_config.py` | Optimized training configuration |

## Verification

To verify experiment authenticity:
1. All training code is provided
2. Complete training history in `results.json`
3. Exact configurations in `config.json`
4. Multi-seed results showing consistency

## Acknowledgments

Base architecture from [Spikformer](https://github.com/ZK-Zhou/spikformer).
SNN framework: [SpikingJelly](https://github.com/fangwei123456/spikingjelly).

## License

MIT License
