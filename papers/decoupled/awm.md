# AWM — Advantage Weighted Matching

> Notation: follows [NOTATION.md](../NOTATION.md) §3 (flow matching convention). Flow model: $v_\theta$; forward path $x_t = (1-t)x_0 + t\epsilon$; clean velocity target $u_t = x_0 - \epsilon$. Advantage $\hat A^{(i)}$ per [NOTATION.md §5](../NOTATION.md). Timestep weight: $w(t)$.

| Field | Value |
|---|---|
| **arXiv** | [2509.25050](https://arxiv.org/abs/2509.25050) |
| **Submitted** | 2025-09-29 |
| **Venue** | — (preprint) |
| **Authors** | Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, Zhi-Ming Ma |
| **Paradigm** | **Decoupled** — clean-target flow matching loss reweighted by advantage; no SDE, no importance ratio |
| **Cites** | DDPO (2305.13301), FlowGRPO (2505.05470), flow matching, LLM pretraining alignment |

---

## Motivation

AWM starts from a theoretical diagnosis: **DDPO is implicitly doing noisy-target flow matching**. Because DDPO's policy gradient uses the denoised sample $x_{t-1}$ (which is stochastic) as the matching target, it inherits high variance that diverges from the clean-target pretraining objective. AWM replaces the noisy target with the clean pretraining target, keeping the advantage weighting — achieving the same RL signal at much lower variance, with no SDE required.

---

## Setting

- **Model**: flow matching model $v_\theta$; reference policy $v_{\theta_\text{ref}}$ (initial checkpoint, kept frozen or updated periodically).
- **Online**: generate $N$ images per prompt each iteration with **any ODE sampler**.
- **Reward**: $r^{(i)} = r(x_0^{(i)}, c) \in \mathbb{R}$; normalised into advantage $\hat A^{(i)}$.

---

## Sampling (inference)

Any deterministic ODE sampler from $t=1$ to $t=0$:
$$x_{t-\Delta t} = x_t - v_\theta(x_t, t, c)\,\Delta t$$

No SDE required at any stage. The clean $x_0^{(i)}$ produced by the ODE is used directly as the matching target.

---

## Key theoretical result — DDPO = noisy-target matching

### Theorem (Xue et al. 2025, informal)

For a Gaussian diffusion/flow model, the DDPO policy gradient update at timestep $t$ is equivalent to:

$$\nabla_\theta \mathcal{L}_\text{DDPO} \propto r \cdot \nabla_\theta \left\|v_\theta(x_t, t, c) - \underbrace{(x_{t-1} - \epsilon)}_{\text{noisy target}}\right\|^2$$

where $x_{t-1}$ is a *stochastic* sample from the policy at step $t-1$.

**Compare to the pretraining loss** (flow matching), which uses the **clean** target:
$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t,\epsilon,x_0}\!\left[\|v_\theta(x_t, t, c) - \underbrace{(x_0 - \epsilon)}_{u_t,\,\text{clean target}}\|^2\right]$$

The gap:
$$\underbrace{x_{t-1} - \epsilon}_\text{DDPO noisy target} = \underbrace{x_0 - \epsilon}_{u_t} + \underbrace{(x_{t-1} - x_0)}_\text{noise residual}$$

The noise residual $(x_{t-1} - x_0)$ has variance $\propto (1-t)^2 \|\Delta v_\theta\|^2$, which is **non-zero** whenever the model prediction deviates from the data manifold. This variance inflates gradients and causes instability.

---

## Reward and advantage calculation

Group-relative advantage (same as GRPO):
$$\hat A^{(i)} = \frac{r^{(i)} - \overline r}{\text{std}(\{r^{(j)}\}) + \delta}$$

where the group contains all $N$ images generated for the same prompt $c$.

---

## Training objective

### AWM loss

Replace the noisy DDPO target with the clean velocity target $u_t = x_0 - \epsilon$, weighted by advantage:

$$\boxed{
\mathcal{L}_\text{AWM}(\theta) = \mathbb{E}_{t,\epsilon}\!\left[\frac{1}{N}\sum_{i=1}^N w(t)\,\hat A^{(i)}\,\left\|v_\theta(x_t^{(i)}, t, c) - u_t^{(i)}\right\|^2\right]
}$$

where:
$$x_t^{(i)} = (1-t)\,x_0^{(i)} + t\,\epsilon^{(i)}, \quad u_t^{(i)} = x_0^{(i)} - \epsilon^{(i)}$$
$$w(t) \geq 0 \quad \text{— timestep weighting schedule}$$

**Effect of advantage sign**:
- $\hat A^{(i)} > 0$ (above-mean reward): the loss pushes $v_\theta$ toward the clean target for this image — reinforcing the generation of similar images.
- $\hat A^{(i)} < 0$ (below-mean reward): the loss pushes $v_\theta$ *away* from this image's target — suppressing the generation of similar images.

### Timestep weighting $w(t)$

The default is $w(t) = 1$ (uniform), matching the standard flow matching pretraining schedule. Optionally:
- Cosine or logit-normal weighting to upweight intermediate $t$ (where the model has the most to learn).
- Min-SNR clipping (imported from DDPM practice) to prevent high-$t$ steps from dominating.

### Connection to LLM alignment

In LLMs, pretraining and RLHF share the same cross-entropy objective:
$$\mathcal{L}_\text{SFT} = -\mathbb{E}[\log \pi(y|x)], \quad \mathcal{L}_\text{PPO} = -\mathbb{E}[\hat A \cdot \log \pi(y|x)]$$

AWM establishes the **exact analogue for diffusion models**:
$$\mathcal{L}_\text{FM} = \mathbb{E}[\|v_\theta - u_t\|^2], \quad \mathcal{L}_\text{AWM} = \mathbb{E}[\hat A \cdot \|v_\theta - u_t\|^2]$$

The only difference is the advantage weighting — the base objective is identical. This means AWM post-training is maximally compatible with the pretraining optimiser, learning rate schedule, and batch size.

### Connection to AWR

Advantage-Weighted Regression (AWR, Peng et al. 2019) in offline RL:
$$\mathcal{L}_\text{AWR}(\theta) = \mathbb{E}_{(s,a) \sim \mathcal{D}}\!\left[\exp\!\left(\hat A(s,a)/\lambda\right) \cdot \left\|-\nabla_\theta \log \pi_\theta(a|s)\right\|^2\right]$$

AWM is AWR applied to diffusion: replacing $\exp(\hat A/\lambda)$ with the linear advantage, and replacing the log-likelihood matching loss with the flow matching MSE.

---

## Training algorithm

```
Input: pretrained v_θ, reward r, prompt dist p_c, group size N, weight w(t)
Repeat:
  1. Sample prompts {c_j}
  2. For each c_j, generate N images via ODE (any fast sampler):
       x_0^(1),...,x_0^(N) ~ ODE_θ(c_j)
  3. Compute rewards: R^(i) = r(x_0^(i), c_j)
  4. Compute group advantages:
       Â^(i) = (R^(i) - mean({R^(j)})) / (std({R^(j)}) + δ)
  5. For each training batch (t, ε):
       t ~ Uniform[0,1];   ε^(i) ~ N(0,I)   [per image]
       x_t^(i)  = (1-t)·x_0^(i) + t·ε^(i)   ← forward noising
       u_t^(i)  = x_0^(i) - ε^(i)            ← clean target
       L = mean_i [ w(t) · Â^(i) · ||v_θ(x_t^(i), t, c_j) - u_t^(i)||² ]
  6. θ ← θ - η ∇_θ L
  (No reference policy update; no importance ratio; no SDE step.)
```

---

## Comparison to related methods

| Method | Objective | Target | SDE needed | Relation to pretraining |
|---|---|---|---|---|
| DDPO | Policy gradient IS | Noisy $x_{t-1}$ (implicit) | Yes | Diverges (noisy target) |
| FlowGRPO | PPO-clipped IS ratio | Per-step $\mu_\theta$ | Yes (ODE→SDE) | Diverges |
| AWM | Advantage-weighted flow matching | Clean $u_t = x_0{-}\epsilon$ | **No** | **Identical base loss** |
| DiffusionNFT | Contrastive implicit policies | Clean $u_t$ (via $v^{\pm}$) | No | Near-identical |
| DGPO | Group Bradley-Terry ELBO | ELBO (per group) | No | Extends Diffusion-DPO |

---

## Results

| Backbone | Benchmark | Result |
|---|---|---|
| SD3.5-M | GenEval | Matches FlowGRPO at **24× speedup** |
| FLUX | PickScore | Comparable or better |
| Both | OCR accuracy | Significant improvement |

Speedup sources:
1. No SDE: ODE sampler with $T$ steps instead of stochastic rollout.
2. Single-$t$ loss: gradient computed at one sampled $t$ per image, not backpropagated through the full denoising chain.

---

## Limitations

- No explicit KL regularisation against the reference policy; relies on learning rate and advantage magnitude to control policy drift.
- On-policy: requires generating images each iteration (cannot use a fixed offline dataset without reweighting).
- Advantage weighting can be unstable when rewards are sparse or heavily skewed; group normalisation partially mitigates this but does not eliminate it.
- Timestep weight $w(t)$ is a hyperparameter; uniform weighting may not be optimal for all tasks.
