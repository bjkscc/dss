# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**dss** (Diffusion Structure Search) is a Python package implementing a variance-preserving generative diffusion model for surface structure discovery. It generates atomic configurations on surfaces by learning to reverse a noising process, guided by an energy potential. Based on the paper by Nikolaj Rønne (2024), GPLv3 licensed.

## Environment & Installation

```
pip install -e .
```

Core dependencies: `numpy`, `torch`, `tensorboard`, `schnetpack` (also pulls in `pytorch_lightning` and `ase`). Python >= 3.5.

## Architecture

The package assembles three concerns into a PyTorch Lightning `VPDiffusion` module via `get_diffusion_model()` (`dss/helpers.py:112`):

### 1. Diffusion process (`dss/diffusion/vp.py`)
`VPDiffusion` is the central `LightningModule`. It implements a linear beta schedule: `β(t) = β_min + t(β_max - β_min)` with `α(t)` as the integrated variance. Training minimizes a score-matching loss (denoising score matching) optionally combined with an energy/forces potential loss (`score_loss + pot_loss`). Sampling uses Euler-Maruyama integration, either plain (`sample()`) or with regressor guidance (`regressor_guidance_sample()`) where the potential model's predicted forces bias the reverse process toward low-energy structures.

Key details:
- Z-confinement restricts atoms to a vertical slab via a truncated normal distribution during sampling
- A binary `mask` excludes substrate atoms from diffusion updates
- Periodic boundary conditions are handled via 27-cell offset lookup (`dss/utils/offsets.py`)

### 2. Score model (`dss/models/score.py`)
`ConditionedScoreModel` wraps a SchNetPack `PaiNN` representation. It appends a sinusoidal time embedding (sin/cos Fourier features, period `2π`) and optional scalar conditioning to the scalar features, then passes `(scalar, vector)` through stacked `GatedEquivariantBlock` layers to produce a 3D vector score per atom. The architecture is SE(3)-equivariant.

### 3. Potential model (`dss/models/potential.py`)
`Potential` extends SchNetPack's `NeuralNetworkPotential`. Shares the same PaiNN representation with the score model. Predicts per-structure energy and per-atom forces. Used during regressor-guided sampling where `(1 - t) * η * forces` steers the reverse process.

### Conditioning (`dss/models/conditionings.py`)
Implements classifier-free guidance style conditioning on scalar properties (e.g., energy). During training, conditioning is randomly dropped with probability `1 - train_prob` so the model learns both conditional and unconditional score prediction, enabling guidance weighting at inference.

### Utilities (`dss/utils/`)
- `TorchNeighborList` — GPU-accelerated neighbor list construction (adapted from TorchANI)
- `EMA` / `EMAOptimizer` / `EMACheckpoint` — exponential moving average of model weights with checkpoint integration
- `TruncatedNormal` — truncated normal distribution used for z-confinement during sampling

## Data pipeline

`get_dataset()` (`dss/helpers.py:2`) takes ASE atoms, repeats them to create supercells, attaches energy/forces/mask/z_confinement properties, and wraps them in a SchNetPack `AtomsDataModule` with a 90/10 train/val split. The neighbor list transform and `CastTo32` are applied as data transforms.
