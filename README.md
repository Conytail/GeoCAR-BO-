# GeoCAR-BO: Geometric Equivariant Learning Meets Orthogonal Bayesian Optimization

> **Accelerating CAR-T Cell Design in Automated BioFoundries**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org)

A closed-loop **Active Learning** framework that integrates **Geometric Deep Learning** with **Batch Bayesian Optimization** to accelerate CAR-T cell engineering in automated BioFoundries. The framework achieves **3× faster convergence** to the Pareto frontier compared to sequence-based baselines.

---

## 🧬 Background

Engineering Chimeric Antigen Receptor (CAR) T-cells holds enormous promise for cancer immunotherapy, but the combinatorial design space exceeds **10²⁰ variants** — far beyond the capacity of brute-force screening.

Traditional approaches suffer from two key limitations:

1. **Sequence blindness**: CNNs and Transformers treating proteins as 1D text miss critical *structural epistasis* — non-linear interactions between residues that are distant in sequence but proximal in 3D space.
2. **Batch redundancy**: Standard Bayesian Optimization picks one-at-a-time; naive top-*k* extension causes "mode collapse" where all 96 robotic wells test nearly identical candidates, wasting resources.

**GeoCAR-BO** addresses both with:
- **EquiformerV2** — an SE(3)-equivariant GNN that learns from AlphaFold2-predicted 3D structures
- **MOBO-OSD** — Multi-Objective Bayesian Optimization with Orthogonal Search Directions that fills each 96-well plate with maximally diverse, high-potential candidates

---

## 🏗️ Framework Overview

```
┌─────────────────────────── DBTL Cycle ──────────────────────────────┐
│                                                                       │
│  Design           Build             Test            Learn            │
│ ─────────        ──────           ──────          ──────────        │
│ CAR domain  →  DNA synthesis  →  Cytotoxicity  →  EquiformerV2     │
│ & motif         Lentiviral        Cytokine         (SE(3)-GNN)      │
│ selection       transduction      Exhaustion        ↓                │
│    ↑               96-well         assays       GP Surrogate        │
│    │              plate format                       ↓               │
│    └──────────────────────────────────── MOBO-OSD Selection ────────┘
```

The AI loop:
1. **Structure** → AlphaFold2 predicts 3D structure of each CAR variant
2. **Encode** → EquiformerV2 extracts SE(3)-equivariant geometric embeddings
3. **Surrogate** → Gaussian Process regressor models cytotoxicity & exhaustion
4. **Select** → MOBO-OSD selects the next diverse batch of 96 candidates
5. **Experiment** → Robotic BioFoundry tests the batch → results feed back to step 2

---

## 📊 Key Results

### Predictive Performance (Daniels et al. 2022 dataset, 2,379 CAR variants)

| Encoder | Pearson *r* (test) | Rel. Training Time |
|---|---|---|
| One-hot encoding | 0.52 | 1× (baseline) |
| ESM-2 language model | 0.69 | ~4× |
| **EquiformerV2 (Ours)** | **0.78** | ~12× |

### Convergence Speed

| Method | Rounds to 95% Hypervolume | Experiments |
|---|---|---|
| Random Search | >10 rounds | >960 |
| Sequence-based GP-UCB | 9 rounds | 864 |
| **Geometric + MOBO-OSD (Ours)** | **3 rounds** | **288** |

→ **3× fewer experiments** to reach the same Pareto frontier coverage.

### Batch Diversity (MOBO-OSD vs. baselines)

| Strategy | Clusters Covered | LogDet Score |
|---|---|---|
| Top-*k* UCB | 1 (85% concentrated) | baseline |
| Random Sampling | Uniform | — |
| **MOBO-OSD (Ours)** | **6–8 distinct clusters** | **+40%** |

### Wet-Lab Validation (preliminary, executed via CRO)

| Metric | Geometric-Optimized | Sequence-Based | CD19-CAR ctrl |
|---|---|---|---|
| Cytotoxicity (% lysis, 5:1 E:T) | **82 ± 3** | 65 ± 4 | 70 ± 3 |
| IFN-γ (pg/mL) | **1200 ± 150** | 480 ± 60 | 560 ± 70 |
| IL-2 (pg/mL) | **800 ± 90** | 310 ± 40 | 370 ± 50 |
| Exhaustion (PD-1/TIM-3 %) | **15 ± 2** | 35 ± 4 | 30 ± 3 |
| Median survival (NSG mice) | **45 days** | 28 days | 32 days |

*p-values: cytotoxicity & cytokines p<0.01; exhaustion & survival p<0.05/0.001 (ANOVA + Tukey; log-rank)*

---

## 🗂️ Repository Structure

```
GeoCAR-BO/
├── BioFoundry_ActiveLearning_Colab.py   # Main pipeline (Colab-ready)
│
├── equiformer_v2/                       # EquiformerV2 model code
│   ├── nets/equiformer_v2/              # Model architecture
│   │   ├── equiformer_v2_oc20.py        # Main model class
│   │   ├── transformer_block.py         # Equivariant attention blocks
│   │   ├── so3.py                       # SO(3) irreducible representations
│   │   ├── so2_ops.py                   # SO(2) operations
│   │   ├── activation.py, drop.py       # Custom layers
│   │   └── layer_norm.py                # Equivariant layer norm
│   ├── oc20/trainer/                    # OC20 trainer
│   ├── main_oc20.py                     # Training entry point
│   ├── engine.py                        # Training engine
│   ├── create_lmdb_dataset.py           # Convert PDB → LMDB format
│   └── config.yml / gpu_config.yml      # Training configurations
│
├── ocp/                                 # Open Catalyst Project framework
│   ├── ocpmodels/                       # Model registry & utilities
│   ├── configs/                         # Pre-built configs
│   └── main.py                          # OCP training entry
│
├── dbgoodman-tcsl-lenti-e418b59/        # CAR library data processing
│   ├── src/py/                          # Python processing scripts
│   ├── src/r/                           # R analysis scripts
│   └── process_data.py                  # Data preprocessing pipeline
│
├── BUGFIX_OCP_Installation.md           # OCP installation troubleshooting
├── BUGFIX_submitit.md                   # submitit import fix guide
└── README_Colab_Usage.md                # Detailed Colab usage guide
```

---

## 🚀 Quick Start (Google Colab)

The entire pipeline is packaged in a single Colab-ready script. Open it in Google Colab:

```python
# 1. Clone this repository
!git clone https://github.com/Conytail/GeoCAR-BO-.git
%cd GeoCAR-BO-

# 2. Run the pipeline script (auto-installs all dependencies)
%run BioFoundry_ActiveLearning_Colab.py
```

The script will guide you through all steps automatically.

---

## 📋 Step-by-Step Pipeline

### Step 0 — GPU Check & Environment
```python
# Auto-detects GPU and sets optimal batch size:
# A100 → batch_size=16, lmax=[4]
# V100 → batch_size=8,  lmax=[4]
# T4   → batch_size=4,  lmax=[2]   ← Critical: T4 cannot handle lmax=4
```

### Step 1 — Install Dependencies
```bash
# Installs PyTorch 2.1.0 + PyG (CUDA 12.1) + OCP + BioPython + BoTorch
# Key: scipy==1.13.1 required for sph_harm compatibility
```

### Step 2 — Prepare Data
Upload your LMDB dataset (AlphaFold2-predicted structures as atomic graphs) to Google Drive:
```
/content/drive/My Drive/BioFoundry/data/
    ├── train.lmdb
    └── val.lmdb
```

Data format: Each LMDB entry is a `torch_geometric.data.Data` object:
- `pos` — atom coordinates (N, 3)
- `atomic_numbers` — atom types (N,)
- `y` — target score (cytotoxicity)
- `sid` — sample ID (CAR variant name)

### Step 3 — Convert Structures (if starting from PDB)
```bash
python equiformer_v2/create_lmdb_dataset.py \
    --src ./pdb_dataset \
    --out ./data/train.lmdb
```

### Step 4 — Train EquiformerV2
```bash
python equiformer_v2/main_oc20.py \
    --config-yml equiformer_v2/gpu_config.yml \
    --mode train \
    --run-dir ./checkpoints
```
Training time: ~2h (A100) / ~3h (V100) / ~6h (T4) for 50 epochs.

### Step 5 — Extract Geometric Embeddings
The script uses `register_forward_hook` to capture the latent feature vector **before** the energy prediction head:
```python
# Hook captures embeddings from the final norm layer
hook_handle = model.energy_block.register_forward_pre_hook(...)
embeddings = {}  # {sample_id: np.array of shape (d,)}
```

### Step 6 — Active Learning Loop (MOBO-OSD)
```python
optimizer = BatchDiversityOptimizer(
    embeddings_dict=embeddings,
    initial_scores=initial_scores,   # {sample_id: score}
    beta=2.0                          # UCB exploration parameter
)
optimizer.fit()

# Select next batch for experiments
next_batch = optimizer.select_batch_diverse(
    batch_size=96,          # Match your 96-well plate format
    diversity_weight=0.5    # 0=pure exploitation, 1=pure diversity
)
```

### Step 7 — Update & Iterate
```python
# After wet-lab results come back:
optimizer.update(new_experimental_results)
optimizer.fit()
next_batch = optimizer.select_batch_diverse(batch_size=96)
# Repeat until convergence
```

---

## ⚙️ Configuration

Key hyperparameters in `equiformer_v2/config.yml`:

```yaml
model:
  num_layers: 4          # Equivariant attention layers (paper uses 6)
  sphere_channels: 64    # Feature channels per irrep
  lmax_list: [4]         # Max angular momentum (reduce to [2] for T4 GPU)
  max_radius: 12.0       # Neighbor cutoff radius (Å)
  max_neighbors: 20      # Graph connectivity

optim:
  lr_initial: 0.001
  optimizer: AdamW
  max_epochs: 50
  scheduler: ReduceLROnPlateau
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| CUDA Out of Memory | Reduce `batch_size` and set `lmax_list: [2]` |
| `sph_harm` error | Pin `scipy==1.13.1` |
| `submitit` import error | See `BUGFIX_submitit.md` |
| OCP installation fails | See `BUGFIX_OCP_Installation.md` |
| Hook captures wrong layer | Print `model.named_modules()` to find correct layer name |
| LMDB read is slow | Copy LMDB to local disk (not Google Drive mount) before training |

---

## 📦 Dependencies

```
torch==2.1.0
torch-geometric
torch-scatter, torch-sparse, torch-cluster
lmdb
biopython
ase
e3nn
scipy==1.13.1
scikit-learn
matplotlib, seaborn
wandb / tensorboard
```

Install via the setup script in `ocp/`:
```bash
conda env create -f ocp/env.gpu.yml
conda activate ocp
pip install -e ocp/
pip install -e equiformer_v2/
```

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{zhang2025geocarbo,
  title   = {Geometric Equivariant Learning Meets Orthogonal Bayesian Optimization:
             Accelerating CAR-T Cell Design in Automated BioFoundries},
  author  = {Zhang, Qifan},
  school  = {School of Information Technology, Monash University Malaysia},
  year    = {2025}
}
```

Key references used in this work:

- **Dataset**: Daniels et al. (2022). *Decoding CAR T cell phenotype using combinatorial signaling motif libraries.* Science 378, 1194–1200. [DOI](https://doi.org/10.1126/science.abq0225)
- **AlphaFold2**: Jumper et al. (2021). *Highly accurate protein structure prediction with AlphaFold.* Nature 596, 583–589.
- **EquiformerV2**: Liao et al. (2024). *EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations.* ICLR 2024.
- **MOBO-OSD**: Ngo et al. (2025). *Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions.* NeurIPS 2025.

---

## 📁 Data Availability

- **CAR constructs**: Deposited in Addgene (plasmid IDs: 197081–197093, available upon publication)
- **Raw wet-lab data**: Zenodo DOI: [10.5281/zenodo.18731187](https://doi.org/10.5281/zenodo.18731187) (activated upon publication)
- **Daniels et al. dataset**: [Science 2022](https://doi.org/10.1126/science.abq0225)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](equiformer_v2/LICENSE) for details.

The EquiformerV2 model code is adapted from [atomicarchitects/equiformer_v2](https://github.com/atomicarchitects/equiformer_v2) and the [Open Catalyst Project](https://github.com/Open-Catalyst-Project/ocp).
