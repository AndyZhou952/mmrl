# SRPO — Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference

> Notation: follows [NOTATION.md](../NOTATION.md). Target model: FLUX.1-dev (flow matching). Flow convention: $x_t = (1-t)x_0 + t\epsilon$, so $\alpha_t = 1-t$, $\sigma_t = t$. Local symbol: $\epsilon_\text{gt}$ — the ground-truth noise for a specific image, held fixed as a training-time prior.

| Field | Value |
|---|---|
| **arXiv** | [2509.06942](https://arxiv.org/abs/2509.06942) |
| **Submitted** | 2025-09-08 |
| **Venue** | — (preprint) |
| **Authors** | Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang, Donghao Li, Chunyu Wang, Qinglin Lu, Yansong Tang |
| **Affiliation** | Tencent Hunyuan; CUHK-Shenzhen; Tsinghua University |
| **GitHub** | https://github.com/Tencent-Hunyuan/SRPO |
| **Paradigm** | **Direct Preference** — no SDE, no multi-step denoising gradients; single-step closed-form recovery |
| **Cites** | DDPO, FlowGRPO, DanceGRPO, ReFL, DRaFT |
| **Cited by** | HunyuanImage 3.0 pipeline (MixGRPO → SRPO → ReDA stages) |

---

## Context

SRPO introduces two independent innovations that can be used separately or together: (1) **Direct-Align**, a training procedure that aligns the model to a reward using a closed-form noise-recovery step instead of multi-step rollouts; and (2) the **SRPO Reward**, a self-normalising relative score that does not require periodic re-training of the reward model. Both target FLUX.1-dev (flow matching), and both are entirely direct-preference — no SDE sampler, no GRPO importance ratio, no KL divergence against a frozen reference.

---

## Problem 1 — Backpropagating reward gradients through $T$ denoising steps is expensive and leaves early steps unaligned

**Issue**: Direct reward alignment methods (ReFL, DRaFT) backpropagate through the multi-step ODE trajectory: $x_T \to x_{T-1} \to \cdots \to x_0 \to r$. Memory and compute scale linearly with $T$. In practice, only the last few steps are included in the computation graph, leaving early high-noise steps — which control the global structure of the image — completely unaligned to the reward.

**Idea — Direct-Align**: Exploit the known noise prior. In flow matching, the forward process places $x_t$ on a linear path between the image $x_0$ and a fixed noise sample $\epsilon_\text{gt}$:

$$x_t = (1-t)x_0 + t\epsilon_\text{gt}$$

If $\epsilon_\text{gt}$ is known (fixed at the start of training for each image), the clean image can be recovered **in one step**:

$$\hat{x}_0 = \frac{x_t - t\epsilon_\text{gt}}{1-t}$$

This uses the same Tweedie formula as [NOTATION.md §3](../NOTATION.md), but substitutes the **known** $\epsilon_\text{gt}$ instead of the network's prediction — making recovery exact and bypassing multi-step denoising entirely.

**Why this works**: The gradient $\partial r / \partial \theta$ passes through only **one network call** $v_\theta(x_t, t, c)$. There is no rollout, no iterative denoising chain, no trajectory storage. The reward gradient flows back through a single velocity prediction, making this $T\times$ cheaper than full-chain backpropagation. Because $t$ is sampled uniformly over $[0,1]$, every noise level (including early high-noise steps) participates in training — all timesteps are aligned, not just the last few.

**Result**: Direct-Align gives SRPO **75× greater training efficiency than DanceGRPO** (§4.3), fine-tuning FLUX.1-dev to convergence in **~10 min on 32 H20 GPUs** (Fig. 1) while aligning all timesteps. On its own, though, it is not enough — the Fig. 9(d) ablation shows removing it "reduced realism and increased vulnerability to reward hacking," and Direct-Align alone reaches only **5.9% realism**, which Problem 2's relative reward lifts to 38.9%.

### Training procedure (Direct-Align)

1. Fix a noise prior $\epsilon_\text{gt} \sim \mathcal{N}(0,I)$ per image at the start of training (held constant).
2. At each step: sample $t \sim \mathrm{Uniform}[0,1]$; construct $x_t = (1-t)x_0 + t\epsilon_\text{gt}$.
3. One network forward pass: compute $v_\theta(x_t, t, c)$.
4. Closed-form recovery: $\hat{x}_0 = (x_t - t\epsilon_\text{gt}) / (1-t)$.
5. Score: $r(\hat{x}_0, c)$; backpropagate through steps 4→3→$\theta$.

Multiple timesteps are combined with a discounting factor that down-weights the very high-noise end:

$$\mathcal{L}_\text{DA}(\theta) = -\mathbb{E}_t\left[\gamma^{T-t}r\left(\hat{x}_0(x_t, \epsilon_\text{gt}), c\right)\right]$$

### Inversion regularisation

To prevent overfitting to the reward at late (low-noise) timesteps, SRPO adds an **inversion branch**: a gradient step in the direction of noise injection ($+\epsilon_\text{gt}$). This penalises solutions that exploit the mapping for late $t$ without genuinely improving image quality. The regularisation acts as an implicit anchor without requiring a frozen reference model or KL term.

---

## Problem 2 — Absolute reward models drift out of distribution as the policy improves

**Issue**: Reward models trained on a fixed dataset (HPSv2.1, PickScore) assign scores relative to a fixed distribution. As the policy improves, the generated images leave the training distribution of the reward model, causing the reward signal to become unreliable. Periodic re-training of the reward model is expensive and introduces a moving-target dynamic.

**Idea — SRPO Reward (Semantic Relative Preference)**: Instead of an absolute score, compute a **relative score** between a positive and a negative text condition applied to the same generated image:

$$r_\text{SRP}(x_0, c) = f_\text{img}(x_0)^\top (C_+ - C_-)$$

where:
- $f_\text{img}(x_0)$: image embedding (from HPSv2.1, PickScore, or CLIP)
- $C_+ = \mathrm{embed}(c_+)$: embedding of a *positive* condition, e.g., *"a photo-realistic image"*
- $C_- = \mathrm{embed}(c_-)$: embedding of a *negative* condition, e.g., *"an AI-generated image"*

**Why this works**: The reward is a **difference** between two alignment scores. If the policy improves by becoming more photorealistic, $f_\text{img}(x_0)^\top C_+$ increases while $f_\text{img}(x_0)^\top C_-$ decreases — the difference magnifies the signal. Crucially, the relative score is computed entirely by the frozen reward model's embedding function: no additional training is needed. Changing the optimisation target (e.g., from photorealism to oil-painting style) requires only swapping $(c_+, c_-)$.

**Result**: The relative reward drives SRPO's headline human-evaluation gains on FLUX.1-dev — excellent-rate for **Realism 8.2% → 38.9%** (3.7×), **Aesthetics 9.8% → 40.5%** (3.1×), **Overall 5.3% → 29.4%** (Tab. 1, Fig. 4) — and "effectively prevents reward hacking" vs absolute multi-reward setups (Fig. 7). Benchmark scores: Aesthetic 6.194, PickScore 23.040, ImageReward 1.118, HPS 0.289.

| Property | Effect |
|---|---|
| No reward model re-training | Score adapts to policy shifts via the difference structure |
| Built-in negative regularisation | Negative branch penalises "AI-generated" aesthetics |
| Model-agnostic | Works with any image-text embedding model |
| Adjustable direction | New aesthetic objective = new $(c_+, c_-)$ pair |

---

## Combined Training Objective

$$\boxed{\mathcal{L}_\text{SRPO}(\theta) = -\mathbb{E}_{c,t,\epsilon_\text{gt}}\left[\gamma^{T-t}r_\text{SRP}\left(\hat{x}_0(x_t, \epsilon_\text{gt}), c\right)\right] + \lambda\mathcal{L}_\text{inv}(\theta)}$$

No KL divergence. No frozen reference model. No SDE sampler required.

---

## Algorithm

```
Input: pretrained v_θ (FLUX.1-dev), reward embedder f_img, text pairs (c_+, c_-),
       discount γ, regularisation weight λ
Initialize: for each training image x_0, fix ε_gt ~ N(0,I)

Repeat:
  1. Sample prompt c and training image x_0 (or generate one)
  2. Sample t ~ Uniform[0,1]
  3. Construct noised image:  x_t = (1-t)·x_0 + t·ε_gt
  4. Forward pass:  v_θ_out = v_θ(x_t, t, c)
  5. Closed-form recovery:  x̂_0 = (x_t - t·ε_gt) / (1-t)      ← one step, no rollout
  6. SRPO reward:
       r = f_img(x̂_0)ᵀ (C_+ - C_-)                             ← no reward model training
  7. Reward loss:  L_DA = -γ^(T-t) · r
  8. Inversion regularisation:
       x_t^inv = (1-t)·x_0 + t·(−ε_gt)                         ← noise injection direction
       L_inv = ||v_θ(x_t^inv, t, c) - target||²
  9. Total:  L = L_DA + λ·L_inv
  10. θ ← θ - η ∇_θ L
```

---

## Comparison to Other Methods

| Aspect | SRPO | FlowGRPO | MixGRPO | AWM | DGPO |
|---|---|---|---|---|---|
| SDE required | **No** | Yes | Partial | No | No |
| Gradient path | 1 network call / $t$ | $T$ SDE steps | $w$ SDE steps | Advantage-weighted loss | ELBO preference |
| Reference model | **No** (inversion reg.) | Yes | Yes | Yes | Yes |
| Reward re-training | **No** (relative reward) | Scalar ORM | Scalar ORM | Scalar ORM | Preference pairs |
| Training time (32× H20) | $<$10 min | Hours | ~50% of FlowGRPO | — | ~5% of FlowGRPO |
| Target model | FLUX.1-dev | SD3.5-M, FLUX | HunyuanVideo | Various | Various |

Reported: ranked #1 on Artificial Analysis Leaderboard for open-source T2I (October 2025); 75× faster than DanceGRPO full-SDE baseline.

---

## Limitations

- Recovery accuracy degrades at very high noise levels ($t \approx 1$) — mitigated by the discount $\gamma^{T-t}$ which down-weights early timesteps.
- The SRPO reward is only as discriminative as the text encoder's ability to separate $(c_+, c_-)$ semantically; aesthetic directions poorly captured by CLIP embeddings will produce weak signals.
- The fixed noise prior $\epsilon_\text{gt}$ per image means each training image follows a fixed noising trajectory — limits gradient diversity compared to stochastic SDE rollouts.
