# DDPO — Training Diffusion Models with Reinforcement Learning

> Notation: follows [NOTATION.md](../NOTATION.md) §7 (MDP) and §4 (policy). Uses DDPM convention: $\epsilon_\theta$, discrete $t \in \{1,\ldots,T\}$.

| Field | Value |
|---|---|
| **arXiv** | [2305.13301](https://arxiv.org/abs/2305.13301) |
| **Submitted** | 2023-05-22 (revised 2024-01-04) |
| **Venue** | ICLR 2024 |
| **Authors** | Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, Sergey Levine |
| **GitHub** | https://github.com/kvablack/ddpo-pytorch |
| **Paradigm** | Coupled — per-step log-prob required; must use stochastic DDPM sampler |
| **Cites** | PPO, REINFORCE, Ho et al. 2020 (DDPM), Ouyang et al. 2022 (RLHF) |
| **Cited by** | FlowGRPO, DanceGRPO, MixGRPO, CPS, DiffusionNFT, AWM, DGPO, GRPO-Guard |

---

## Motivation

Diffusion models maximise data likelihood, but downstream objectives (aesthetics, prompt alignment, compressibility) cannot be expressed as a likelihood. Reward-weighted likelihood is high-variance and requires white-box access. DDPO reformulates denoising as an MDP so that standard policy gradient algorithms can optimise arbitrary black-box reward functions.

---

## Setting

- **Model**: DDPM with noise predictor $\epsilon_\theta(x_t, t, c)$.
- **Reverse process** (policy): $\pi_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t,t,c),\, \tilde\beta_t I)$ where
$$\mu_\theta(x_t,t,c) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\hat x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t, \qquad \hat x_0 = \frac{x_t - \sigma_t\,\epsilon_\theta(x_t,t,c)}{\sqrt{\bar\alpha_t}}$$
- **Reward**: $r(x_0, c) \in \mathbb{R}$, black-box, evaluated once per trajectory.

---

## Sampling (inference)

Standard DDPM ancestral sampling:

$$x_{t-1} = \mu_\theta(x_t, t, c) + \tilde\beta_t^{1/2}\, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,I), \quad t = T, T{-}1, \ldots, 1$$

starting from $x_T \sim \mathcal{N}(0,I)$. At $t=0$ the sample $x_0$ is returned and evaluated.

---

## Reward calculation

Given $x_0$ and prompt $c$, any scalar reward model $r : \mathbb{R}^d \times \mathcal{C} \to \mathbb{R}$:
- Examples in the paper: JPEG compressibility, LAION aesthetic score, BLIP-v2 caption similarity, human preference score.
- Reward need not be differentiable w.r.t. $x_0$.

---

## Training objectives

### DDPO$_\text{SF}$ — REINFORCE ("score function" estimator)

Unbiased but high-variance. Backprop through $\log \pi_\theta$ at every denoising step:

$$\mathcal{L}_\text{SF}(\theta) = -\mathbb{E}_{c, x_T, \{x_{t-1}\}}\!\left[\sum_{t=1}^{T} \log \pi_\theta(x_{t-1} \mid x_t, c) \cdot r(x_0, c)\right]$$

Since $\tilde\beta_t$ is fixed, $\log \pi_\theta(x_{t-1} \mid x_t, c) \propto -\|x_{t-1} - \mu_\theta(x_t,t,c)\|^2 / (2\tilde\beta_t)$, giving an MSE-like per-step gradient.

### DDPO$_\text{IS}$ — Importance-sampling PPO (recommended)

Collect a trajectory with frozen $\theta_\text{old}$, then run $K$ gradient steps reusing the same data:

$$\mathcal{L}_\text{IS}(\theta) = -\mathbb{E}\!\left[\sum_{t=1}^{T} \rho_t \cdot r(x_0, c)\right], \quad \rho_t = \frac{\pi_\theta(x_{t-1} \mid x_t, c)}{\pi_{\theta_\text{old}}(x_{t-1} \mid x_t, c)}$$

The IS ratio is tractable because each step is Gaussian:
$$\log \rho_t = \frac{\|x_{t-1} - \mu_{\theta_\text{old}}\|^2 - \|x_{t-1} - \mu_\theta\|^2}{2\tilde\beta_t}$$

A clipped variant (analogous to PPO) applies $\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)$ to bound the update.

**No baseline / advantage normalisation** in the original DDPO (added later in FlowGRPO). All steps of a trajectory share the same reward scalar.

### KL regularisation (optional)

$$\mathcal{L}_\text{KL}(\theta) = \beta\, \mathbb{E}\!\left[\sum_{t=1}^T D_\text{KL}\!\left(\pi_\theta(\cdot \mid x_t, c) \,\|\, \pi_\text{ref}(\cdot \mid x_t, c)\right)\right]$$

---

## Training algorithm

```
Input: reward model r, prompt distribution p_c, pretrained θ
Repeat:
  1. Sample batch of prompts {c_1,...,c_B} ~ p_c
  2. For each c_i, rollout T steps under π_{θ_old}:
       x_T ~ N(0,I);  for t=T,...,1: x_{t-1} ~ π_{θ_old}(·|x_t, c_i)
     Store full trajectory {x_T, x_{T-1},..., x_0}
  3. Compute rewards: R_i = r(x_0^i, c_i)
  4. For K gradient steps:
       Compute ρ_t^i = π_θ(x_{t-1}^i|x_t^i,c_i) / π_{θ_old}(x_{t-1}^i|x_t^i,c_i)
       L = -mean_i [ sum_t clip(ρ_t^i, 1-ε, 1+ε) · R_i ]
       θ ← θ - η ∇_θ L
  5. θ_old ← θ
```

**Memory**: storing all $T$ intermediate $x_t$ is the main cost — $O(T \cdot d)$ per sample.

---

## Relation to GRPO (what FlowGRPO adds)

| Aspect | DDPO | FlowGRPO |
|---|---|---|
| Advantage | Raw reward $r$ (no baseline) | Group-normalised $\hat A^{(i)}$ (variance reduction) |
| Stochasticity | DDPM sampler (naturally stochastic) | ODE→SDE conversion (needed for flow matching) |
| Steps | All $T$ steps | Denoising reduction: subset of steps |
| Model family | DDPM | Flow matching (SD3, FLUX) |

---

## Limitations (addressed by later work)

| Limitation | Addressed by |
|---|---|
| High-variance REINFORCE; no group advantage | FlowGRPO, DanceGRPO |
| $O(T)$ gradient memory; slow | FlowGRPO-Fast, MixGRPO, AWM |
| DDPM objective ≠ pretraining score matching | AWM |
| Cannot use fast ODE samplers | DGPO, AWM, DiffusionNFT |
| No reward hacking protection | GRPO-Guard |
