# TimeCauST

<p align="center">
  <b>Causally Constrained Adversarial Attacks via Granger-structural Confinement</b>
  <br> <i>IEEE DSAA 2026, New Delhi, India</i> 
</p>
<p align="center">
  <!-- <a> href="https://openreview.net/forum?id=Al4OnLoQsp"> -->
  <a>
    <img src="https://img.shields.io/badge/XPLORE-Coming Soon-005995?style=for-the-badge&logo=ieee">
  </a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.13.0-EE4C2C?style=for-the-badge&logo=pytorch&labelColor=dddddd">
  <img src="https://img.shields.io/badge/CUDA-13.2-green?style=for-the-badge&logo=nvidia&labelColor=dddddd">
  <img src="https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&labelColor=dddddd">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge&labelColor=dddddd">
  <img src="https://img.shields.io/badge/Reproducibility-Verified-success?style=for-the-badge&labelColor=dddddd">
</p>
<p align="center">
<a href="https://github.com/ayan-cs/timecaust">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=ayan-cs.TimeCauST?style=for-the-badge">
  </a>
</p>

## 🔭 Overview

**TimeCauST** is a white-box adversarial attack framework for causality-aware multivariate time series (MTS) forecasting models. Given a victim model's learned Granger-causal graph, TimeCauST selectively degrades the forecast of a chosen target channel while minimising perturbation spillover to non-target channels. The attack exploits the causal structure via an iteratively causally-constrained optimisation procedure that enforces both structural feasibility (perturbation confined to causal parents of the target) and $L_\infty$ budget constraints.

**Note**: The framework currently supports CR-VAE as the victim model, with experiments on H´enon dataset.

---
## ✒️ Key Contributions

### *Causally constrained adversarial attack*

**TimeCauST** confines the adversarial perturbation exclusively to the Granger-causal parent set of the designated target channel via a structural projection operator, making it the first attack to exploit a victim model's own learned causal graph as the attack mechanism.

### *Controlled causal intervention protocol*

A three-condition intervention experiment — parent-only, non-parent-only, and cardinality-matched random-channel perturbation — provides causal rather than correlational evidence that attack efficacy genuinely follows the learned causal graph. The ordering $CSS(Parent) > CSS(Random) > CSS(Non-parent)$ holds across all $7$ datasets.

### *Three novel causal evaluation metrics*

These are introduced to quantify, respectively, the selectivity of the attack toward the target channel, structural alignment of the perturbation with the causal graph, and target disruption delivered per unit perturbation budget.

- **Causal Selectivity Score**

  $$\mathrm{CSS} = \frac{\big\| \tilde{y}^{(i)} - \hat{y}^{(i)} \big\|\_{F}}{\big\| \tilde{y}^{(-i)} - \hat{y}^{(-i)} \big\|_{F} + \gamma}$$

- **Causal Graph Alignment**

  $\mathrm{CGA} = \frac{\big\| \delta^{(\mathcal{P}\_i)} \big\|_{F}}{\big\| \delta \big\|\_{F} + \gamma}$

- **Perturbation Efficiency**

  $\mathrm{PE} = \frac{\big\| \tilde{y}^{(i)} - \hat{y}^{(i)} \big\|\_{F}}{\big\| \delta \big\|\_{F} + \gamma}$

---
## 🔬 Methodology

### 1. Causal parent set extraction

Given white-box access to the victim model's learned Granger-causal adjacency matrix $A$, TimeCauST computes the parent set $P_i = \\{j : A_{ij} = 1\\}$ for the designated target channel $i$. No separate causal discovery is performed; the victim's own graph is consumed directly.

### 2. Structural feasibility set and projection

The perturbation $\delta$ is constrained to the feasibility set $F(A, i, \epsilon)$ — zero on all non-parent channels, $L_\infty$-bounded on parent channels. This is enforced as a hard two-stage projection at every iterate: $L_\infty$ clipping on parent channels, followed by structural zeroing of all remaining channels. The projection is exact, closed-form, and costs $O(\tau D)$ per step.

### 3. Three-term constrained objective

TimeCauST maximises a composite objective:
1. $\mathcal{L}_\mathrm{tgt}$, the forecast error on the target channel;
2. $\lambda_1 \cdot \mathcal{L}_\mathrm{nt}$, the deviation of non-target forecasts from the clean baseline;
3. $\lambda_2 \cdot \mathcal{L}_\mathrm{smooth}$, the temporal roughness of the perturbation.

The structural constraint is not a soft penalty in the objective — it is enforced exactly through feasibility set membership.

---
## 🛠️ Execution

### Setup

```bash
git clone git@github.com:ayan-cs/timecaust.git
cd timecaust
```

### Environment I used
```
python==3.14.5
pytorch==2.13.0+cu132
scikit-learn==1.8.0
numpy==2.3.5
```

Everything is configured in `config.py` — there is no command-line interface.

### 1. Generate the dataset

Each dynamical system has its own generator under `data/.` This release ships the Hénon system used in the paper's experiments:
```
python -m data.henon_gen
```
This writes `data/henon/henon_<timesteps>_<dims>.npz` — the raw trajectory, its z-score-standardized copy, and the ground-truth Granger-causal adjacency matrix the CR-VAE surrogate and the attack both consume.

### 2. Train the CR-VAE surrogate (victim model)

```
python run_model.py
```

Reads `config.RUN_DATASETS / config.RUN_SEEDS` (defaults to Hénon, seed 42) and grid-searches `config.CRVAE_PARAM_GRID`. Every combination is trained to early stopping; the one with the lowest validation loss is written to: `artifacts/seed<S>/artifacts_henon/crvae__grid__seed<S>__<timestamp>/best/checkpoint.pt` and promoted as the dataset's selected checkpoint in `artifacts/seed<S>/artifacts_henon/registry.json`.

### 3. Run the TimeCauST attack

```
python run_attack.py
```

Automatically finds the checkpoint trained in Step 2 via `registry.json` — no path to type in by hand. It then runs the causally-constrained PGD attack from `config.ATTACK_PARAM_GRID` against every target channel, under three conditions (parent-only / non-parent-only / random-mask control), and writes per-combination results to: `artifacts/seed<S>/artifacts_henon/attacks/timecaust__grid__seed<S>__<timestamp>/` including the CSS / CGA / PE metrics.

---
## 📭 Contact

<p align="left">
    <a href="https://www.linkedin.com/in/ayanabha-ghosh-cs">
        <img src="https://img.shields.io/badge/Linkedin-Connect-0a66c2?style=for-the-badge">
    </a>
    <img src="https://img.shields.io/badge/Official-p23iot002%40iitj.ac.in-fcebca?style=for-the-badge&logo=gmail&labelColor=dddddd">
    <a href="https://sites.google.com/view/ayanabha">
        <img src="https://img.shields.io/badge/Portfolio-Visit-%2326c7c2?style=for-the-badge&logo=googlechrome&logoColor=%23ffffff">
    </a>
</p>
