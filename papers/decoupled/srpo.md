# SRPO — Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference

> Notation: follows [NOTATION.md](../NOTATION.md). Target model is FLUX.1-dev (flow matching).  
> For flow matching: $x_t = (1-t)\,x_0 + t\,\epsilon$, so $\alpha_t = 1-t$, $\sigma_t = t$.  
> Local symbol: $\epsilon_\text{gt}$ — the ground-truth noise used in the forward pass, held fixed as a per-image training prior.

| Field | Value |
|---|---|
| **arXiv** | [2509.06942](https://arxiv.org/abs/2509.06942) |
| **Submitted** | 2025-09-08 |
| **Venue** | — |
| **Authors** | Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang, Donghao Li, Chunyu Wang, Qinglin Lu, Yansong Tang |
| **Affiliation** | Tencent Hunyuan; CUHK-Shenzhen; Tsinghua University |
| **GitHub** | https://github.com/Tencent-Hunyuan/SRPO |
| **Domain** | Online RL for image generation (FLUX.1-dev) |
| **Paradigm** | Decoupled — no SDE, no multi-step denoising gradients |
| **Cites** | DDPO, FlowGRPO, DanceGRPO, ReFL, DRaFT |
| **Cited by** | HunyuanImage 3.0 pipeline (MixGRPO → SRPO → ReDA) |

---

## Problem Being Solved

Prior direct-reward alignment methods (ReFL, DRaFT) backpropagate reward gradients through the multi-step denoising chain. Two problems arise:

1. **Gradient cost:** Computing gradients through $T$ denoising steps is expensive; in practice only a few late steps are trained, leaving early high-noise steps unaligned.
2. **Reward model staleness:** Capturing fine-grained aesthetic properties (photorealism, lighting) requires continuous offline re-training of the reward model to keep up with the shifting policy distribution.

SRPO introduces two components that address each problem independently: **Direct-Align** eliminates multi-step backpropagation; the **SRPO reward** eliminates the need to re-train the reward model.

---

## Core Innovation 1: Direct-Align

### Key equation

The forward process places $x_t$ on a linear interpolation between $x_0$ and $\epsilon$:

$$x_t = \alpha_t\,x_0 + \sigma_t\,\epsilon_\text{gt}$$

If the noise $\epsilon_\text{gt}$ used to construct $x_t$ is known, the clean image is recovered in **one closed-form step** — no network denoising required:

$$\hat x_0 = \frac{x_t - \sigma_t\,\epsilon_\text{gt}}{\alpha_t}$$

For flow matching (FLUX): $\alpha_t = 1-t$, $\sigma_t = t$, so $\hat x_0 = \dfrac{x_t - t\,\epsilon_\text{gt}}{1-t}$.

This uses the same structure as Tweedie's formula in [NOTATION.md §3](../NOTATION.md), but substitutes the **known** $\epsilon_\text{gt}$ instead of the network's noise prediction, making it exact and gradient-free on the denoising path.

### Training procedure

1. Fix a **noise prior** $\epsilon_\text{gt} \sim \mathcal{N}(0,I)$ per image at the start of training (held constant throughout).
2. At each training step, sample a timestep $t$ uniformly; construct $x_t = (1-t)\,x_0 + t\,\epsilon_\text{gt}$.
3. Apply one network forward pass to obtain $v_\theta(x_t, t, c)$.
4. Recover $\hat x_0$ via the closed-form equation above.
5. Score $\hat x_0$ with the reward model; backpropagate the reward gradient through the single closed-form step to $v_\theta$.

Gradient flows through exactly **one network call** per timestep — no rollout, no iterative denoising. Multiple timesteps are aggregated with a decaying discount:

$$\mathcal{L}_\text{DA}(\theta) = -\mathbb{E}_t\left[\gamma^{T-t}\,r(\hat x_0(x_t, \epsilon_\text{gt}),\,c)\right]$$

where $\gamma \in (0,1]$ down-weights very early (high-noise) timesteps where recovery is less precise.

### Inversion regularization

An additional **inversion branch** applies gradient descent in the direction of noise injection ($+\epsilon_\text{gt}$), penalizing overfitting to the reward in late timesteps. This acts as an implicit regularizer without requiring a separate reference model or KL divergence term.

---

## Core Innovation 2: SRPO Reward (Semantic Relative Preference Optimization)

### Motivation

Absolute reward models (HPSv2.1, PickScore) drift out of distribution as the policy improves, requiring periodic re-training. The SRPO reward computes a **relative score** between a positive and a negative text condition applied to the same generated image, making it self-normalizing:

$$r_\text{SRP}(x_0, c) = f_\text{img}(x_0)^\top C_+ - f_\text{img}(x_0)^\top C_- = f_\text{img}(x_0)^\top (C_+ - C_-)$$

where:
- $f_\text{img}(x_0)$: image embedding from the reward model (HPSv2.1, PickScore, or CLIP)
- $C_+ = \text{embed}(c_+)$: embedding of a **positive** text condition, e.g., *"a photo-realistic image"*
- $C_- = \text{embed}(c_-)$: embedding of a **negative** text condition, e.g., *"an AI-generated image"*

### Properties

| Property | Effect |
|---|---|
| No offline reward re-training | Changing aesthetic direction requires only swapping $(c_+, c_-)$ |
| Built-in regularization | Negative branch penalizes "AI look"; difference is self-normalizing |
| Online adjustment | Reward adapts to the current policy distribution by construction |
| Model-agnostic | Works with any reward model that computes image-text similarity |

---

## Combined Training Objective

$$\mathcal{L}_\text{SRPO}(\theta) = -\mathbb{E}_{c,\,t,\,\epsilon_\text{gt}}\left[\gamma^{T-t}\,r_\text{SRP}\left(\hat x_0(x_t, \epsilon_\text{gt}),\,c\right)\right] + \lambda\,\mathcal{L}_\text{inv}(\theta)$$

where $\mathcal{L}_\text{inv}$ is the inversion regularization term and $\lambda$ balances reward vs. regularization.

**No KL divergence. No frozen reference model. No SDE sampler required.**

---

## Results

On FLUX.1-dev (500 prompts, human evaluation with 10 annotators + 3 domain experts):

| Metric | Baseline (FLUX) | SRPO | Gain |
|---|---|---|---|
| Realism ("excellent" rate) | 8.2% | 38.9% | +3.7× |
| Aesthetic quality | — | — | +3.1× |

Training efficiency: converges in under 10 minutes on 32 H20 GPUs; 75× faster than DanceGRPO (full-SDE baseline).

Ranked #1 on Artificial Analysis Leaderboard for open-source T2I models (October 2025).

---

## Relation to Other Methods in This Repo

| Aspect | SRPO | FlowGRPO | MixGRPO | AWM | DGPO |
|---|---|---|---|---|---|
| SDE required | **No** | Yes | Partial (window) | No | No |
| Gradient path | 1 closed-form step | $T$ SDE steps | $\lvert W\rvert$ SDE steps | Advantage-weighted loss | ELBO preference |
| Reference model | **No** (inversion reg.) | Yes ($\pi_\text{ref}$) | Yes | Yes | Yes |
| Reward model training | **No re-training** (relative reward) | Scalar ORM | Scalar ORM | Scalar ORM | Preference pairs |
| Training time | <10 min / 32 H20 | Hours | ~50% of Flow-GRPO | — | ~20× faster than SDE |
| Target model | FLUX.1-dev | SD3.5-M | HunyuanVideo DiT | Various | Various |

---

## Limitations

- Recovery accuracy degrades at very high noise levels ($t \approx 1$); mitigated by the discount factor $\gamma^{T-t}$.
- SRPO reward is only as discriminative as the text encoder's ability to separate $c_+$ and $c_-$ semantically; out-of-vocabulary aesthetic dimensions are poorly captured.
- The fixed noise prior $\epsilon_\text{gt}$ means each training image follows a fixed noising trajectory — limits diversity of the gradient signal vs. stochastic SDE rollouts.
