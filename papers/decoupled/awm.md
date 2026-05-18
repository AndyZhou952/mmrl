# AWM — Advantage Weighted Matching

> Notation: follows [NOTATION.md](../NOTATION.md) §3 (flow matching). Flow model: $v_\theta$; forward path $x_t = (1-t)x_0 + t\epsilon$; clean velocity target $u_t = x_0 - \epsilon$. Advantage $\hat{A}^{(i)}$ per [NOTATION.md §5](../NOTATION.md). Timestep weight: $w(t) \geq 0$.

| Field | Value |
|---|---|
| **arXiv** | [2509.25050](https://arxiv.org/abs/2509.25050) |
| **Submitted** | 2025-09-29 |
| **Venue** | — (preprint) |
| **Authors** | Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, Zhi-Ming Ma |
| **GitHub** | https://github.com/scxue/advantage_weighted_matching |
| **Paradigm** | **Decoupled** — flow matching MSE reweighted by advantage; no SDE, no importance ratio |
| **Cites** | DDPO (2305.13301), FlowGRPO (2505.05470), flow matching, LLM pretraining alignment |

---

## Context

AWM addresses a conceptual gap that prior work overlooked: the pretraining objective (flow matching MSE) and the RL objective (GRPO / DDPO) are structurally different losses. AWM establishes that this gap is unnecessary — the pretraining loss can be extended to an RL objective by a single modification: multiplying by the group-relative advantage. The result is a method that is maximally compatible with the pretraining optimiser, requires no SDE, and is provably equivalent to DDPO with variance reduction.

---

## Problem — DDPO (and FlowGRPO) implicitly use a noisy target that diverges from pretraining

**Issue**: DDPO's policy gradient update at timestep $t$ is proportional to the reward times the gradient of $\log \pi_\theta(x_{t-1} | x_t, c)$. For a Gaussian policy, this is equivalent to:

$$\nabla_\theta \mathcal{L}_\text{DDPO} \propto r \cdot \nabla_\theta \left\|v_\theta(x_t, t, c) - \underbrace{(x_{t-1} - \epsilon)}_{\text{noisy target}}\right\|^2$$

The target $(x_{t-1} - \epsilon)$ is a *stochastic* sample from the policy at step $t-1$. Compare this to the pretraining flow matching loss, which uses the **clean** velocity target:

$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t,\epsilon,x_0}\!\left[\left\|v_\theta(x_t, t, c) - \underbrace{(x_0 - \epsilon)}_{u_t,\ \text{clean}}\right\|^2\right]$$

The difference is $(x_{t-1} - x_0)$, the noise residual at step $t-1$. This residual has variance $\propto (1-t)^2\|v_\theta - u_t\|^2$, which is non-zero whenever the model prediction deviates from the data manifold. This **inflates gradients** and causes training instability.

**Idea**: Replace the noisy DDPO target with the clean velocity target $u_t = x_0 - \epsilon$, and weight the loss by the group-relative advantage:

$$\mathcal{L}_\text{AWM}(\theta) = \mathbb{E}_{t,\epsilon}\!\left[\frac{1}{N}\sum_{i=1}^{N} w(t)\,\hat{A}^{(i)}\left\|v_\theta(x_t^{(i)}, t, c) - u_t^{(i)}\right\|^2\right]$$

**Why this works**: The clean target eliminates the noise residual variance completely. The advantage weighting provides the RL signal: positive-advantage samples push $v_\theta$ toward the target (reinforcing the behaviour), while negative-advantage samples push $v_\theta$ away from the target (suppressing the behaviour). Crucially, this is the **exact same loss** as flow matching pretraining — but with advantage weighting added. No architectural changes, no SDE, no importance ratio.

---

## The Pretraining Analogy

AWM establishes a precise analogy between language model alignment and diffusion alignment:

| Domain | Pretraining loss | RL-aligned loss |
|---|---|---|
| LLMs | $\mathcal{L}_\text{SFT} = -\mathbb{E}[\log\pi(y|x)]$ | $\mathcal{L}_\text{PPO} = -\mathbb{E}[\hat{A}\cdot\log\pi(y|x)]$ |
| Diffusion | $\mathcal{L}_\text{FM} = \mathbb{E}[\|v_\theta - u_t\|^2]$ | $\mathcal{L}_\text{AWM} = \mathbb{E}[\hat{A}\cdot\|v_\theta - u_t\|^2]$ |

In both cases, the RL loss is the pretraining loss multiplied by the advantage weight. AWM is the natural flow-matching analogue of PPO's policy gradient — not a heuristic, but the unique extension that preserves the pretraining objective structure.

---

## Reward and Advantage

Group-relative advantage (same as GRPO; see [NOTATION.md §5](../NOTATION.md)):

$$\hat{A}^{(i)} = \frac{r^{(i)} - \overline{r}}{\mathrm{std}(\{r^{(j)}\}) + \delta}, \quad r^{(i)} = r(x_0^{(i)}, c)$$

**Effect of advantage sign**:
- $\hat{A}^{(i)} > 0$: $w(t)\hat{A}^{(i)} > 0$ → loss pushes $v_\theta$ **toward** the clean target for this image → reinforces this generation.
- $\hat{A}^{(i)} < 0$: $w(t)\hat{A}^{(i)} < 0$ → loss pushes $v_\theta$ **away** from this image's target → suppresses this generation.

---

## Training Objective

$$\boxed{
\mathcal{L}_\text{AWM}(\theta) = \mathbb{E}_{t,\epsilon}\!\left[\frac{1}{N}\sum_{i=1}^{N} w(t)\,\hat{A}^{(i)}\left\|v_\theta\!\left((1{-}t)x_0^{(i)} + t\epsilon^{(i)},\ t,\ c\right) - (x_0^{(i)} - \epsilon^{(i)})\right\|^2\right]
}$$

where $w(t) \geq 0$ is a timestep weighting schedule (default: $w(t) = 1$, matching the pretraining schedule). No importance ratio, no SDE, no reference policy during the gradient step.

### Timestep weighting

| Schedule | When to use |
|---|---|
| $w(t) = 1$ (uniform) | Default; matches pretraining |
| Cosine / logit-normal | Upweights intermediate $t$ |
| Min-SNR clipping | Prevents high-$t$ steps from dominating |

---

## Algorithm

```
Input: pretrained v_θ, reward r, prompt dist p_c, group size N, weight w(t)
Repeat:
  1. Sample prompts {c_j}
  2. For each c_j, generate N images via any ODE sampler (no SDE):
       x_0^(1),...,x_0^(N) ~ ODE_θ(c_j)
  3. Compute rewards: R^(i) = r(x_0^(i), c_j)
  4. Group advantage:
       Â^(i) = (R^(i) - mean({R^(j)})) / (std({R^(j)}) + δ)
  5. For each training batch (t, ε):
       t    ~ Uniform[0,1]
       ε^(i) ~ N(0,I)  [per image]
       x_t^(i)  = (1-t)·x_0^(i) + t·ε^(i)    ← forward noising
       u_t^(i)  = x_0^(i) - ε^(i)            ← clean velocity target
       L = mean_i [ w(t) · Â^(i) · ‖v_θ(x_t^(i), t, c_j) - u_t^(i)‖² ]
  6. θ ← θ - η ∇_θ L
  (No SDE. No importance ratio. No reference policy lookup.)
```

---

## Comparison to Related Methods

| Method | Target | SDE | Relation to pretraining |
|---|---|---|---|
| DDPO | Noisy $x_{t-1}$ (implicit) | Yes | Diverges |
| FlowGRPO | Per-step SDE mean $\mu_\theta$ | Yes | Diverges |
| AWM | Clean $u_t = x_0 - \epsilon$ | **No** | **Identical base loss** |
| DiffusionNFT | Clean $u_t$ via implicit policies | No | Near-identical |
| DGPO | ELBO over group | No | Extends Diffusion-DPO |

---

## Results

| Backbone | Benchmark | Result |
|---|---|---|
| SD3.5-M | GenEval | Matches FlowGRPO at **24× speedup** |
| FLUX | PickScore | Comparable or better |
| Both | OCR accuracy | Significant improvement |

Speedup sources: (1) ODE sampler instead of SDE rollout; (2) single sampled $t$ per image — no backpropagation through the denoising chain.

---

## Limitations

- No explicit KL regularisation; relies on learning rate and advantage magnitude to control policy drift (large advantages can cause instability).
- On-policy: requires generating images at each iteration (cannot use a fixed offline dataset without importance-reweighting).
- Advantage weighting can be unstable for sparse or heavily-skewed rewards; group normalisation partially mitigates this.
- Timestep weight $w(t)$ is a hyperparameter; uniform weighting may not be optimal for all tasks or reward types.
