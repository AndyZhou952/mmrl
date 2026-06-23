# DanceGRPO — Unleashing GRPO on Visual Generation

> Notation: follows [NOTATION.md](../NOTATION.md). For DDPM: $\epsilon_\theta$, discrete $t$; diffusion coefficient $\eta_t \in [0,1]$. For flow: $v_\theta$, continuous $t \in [0,1]$; SDE coefficient $\varepsilon_t$. Both cases produce Gaussian per-step transitions.

| Field | Value |
|---|---|
| **arXiv** | [2505.07818](https://arxiv.org/abs/2505.07818) |
| **Submitted** | 2025-05-12 (revised 2025-08-28) |
| **Venue** | — (preprint) |
| **Authors** | Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, Ping Luo |
| **GitHub** | https://github.com/XueZeyue/DanceGRPO |
| **Project** | https://dancegrpo.github.io |
| **Paradigm** | **Policy Gradient** — per-step Gaussian log-prob required; SDE-based sampling |
| **Cites** | DDPO, DPOK, GRPO (DeepSeek-R1), SD3, FLUX, HunyuanVideo, SkyReels-I2V |
| **Cited by** | CPS, DiffusionNFT, GRPO-Guard, DGPO |

---

## Context

DanceGRPO is **concurrent** with FlowGRPO (submitted 4 days later, May 2025) and independently arrives at the same ODE→SDE conversion insight. The key difference is scope: where FlowGRPO targets flow matching for T2I only, DanceGRPO aims for a **single unified GRPO framework** that works across DDPM and rectified flow, and across T2I, T2V, and I2V tasks on four different backbones. Readers are expected to understand both DDPM and flow matching training; the core challenge below — why GRPO cannot be applied to deterministic samplers — is the same as in [FlowGRPO](flow_grpo.md), so this document focuses on how DanceGRPO generalises it.

---

## Problem 1 — A single GRPO implementation cannot serve both DDPM and flow matching

**Issue**: DDPM and rectified flow models have structurally different denoising processes — different noise schedules, different parameterisations ($\epsilon_\theta$ vs. $v_\theta$), and different step distributions. Applying GRPO to each would require two separate codebases with two separate SDE derivations and two separate importance-ratio computations. Existing policy-gradient methods (DDPO) only support DDPM and do not generalise to flow matching or video.

**Idea**: Derive a **unified SDE reverse process** under a common formulation that covers both DDPM (including DDIM-like schedules) and rectified flow. Both derivations yield a Gaussian per-step transition, so the GRPO importance ratio and clip objective take the same form in both cases.

**Why this works**: Both DDPM and rectified flow define continuous-time SDEs via the Fokker-Planck equation. For DDPM, the reverse SDE parameterised by stochasticity $\eta_t \in [0,1]$ already takes a Gaussian form. For rectified flow, adding a score-corrected noise term produces an equivalent SDE with the same Gaussian structure. Since the per-step density has the same form — $\mathcal{N}(\mu_\theta, \sigma^2 I)$ — the importance ratio formula is identical, enabling one unified training loop.

**Result**: The single unified loop delivers gains across **all four backbones/modalities**, up to **+181%** over baselines: Stable Diffusion v1.4 HPS-v2.1 **0.239 → 0.365** (+53%), GenEval **0.421 → 0.522** (+24%); FLUX HPS-v2.1 **0.304 → 0.372** (+22%); HunyuanVideo (T2V) motion quality **1.37 → 3.85** (+181%), visual quality +56%; SkyReels (I2V) motion +91% (Tabs. 2–5) — DDPM and flow handled by one importance-ratio formula.

### A. DDPM reverse SDE

The continuous reverse SDE with tunable stochasticity $\eta_t$:

$$d\mathbf{z}_t = \left(f_t\mathbf{z}_t - \frac{1+\eta_t^2}{2}g_t^2\nabla\log p_t(\mathbf{z}_t)\right)dt + \eta_t g_td\mathbf{W}_t$$

where $f_t$, $g_t$ are the forward-process drift and diffusion coefficients, and the score is approximated as $\nabla\log p_t \approx -\epsilon_\theta(\mathbf{z}_t, t, c)/\sigma_t$. Setting $\eta_t = 1$ recovers full DDPM; $\eta_t = 0$ gives deterministic DDIM. For GRPO, $\eta_t > 0$ is required to have a tractable density. Euler-Maruyama discretisation yields:

$$\pi_\theta(\mathbf{z}_{t-1} \mid \mathbf{z}_t, c) = \mathcal{N}\left(\mathbf{z}_{t-1};\ \mu_\theta^\text{DDPM}(\mathbf{z}_t, t, c),\ \eta_t^2 g_t^2 \Delta t I\right)$$

### B. Rectified flow reverse SDE

The flow ODE $d\mathbf{z}_t = v_\thetadt$ is converted to an SDE by adding score-corrected noise (same derivation as FlowGRPO):

$$d\mathbf{z}_t = \left(v_\theta(\mathbf{z}_t, t, c) - \frac{\varepsilon_t^2}{2}\nabla\log p_t(\mathbf{z}_t)\right)dt + \varepsilon_td\mathbf{W}_t$$

Score approximation via Tweedie: $\nabla\log p_t(\mathbf{z}_t) \approx (\hat{\mathbf{z}}_0 - \mathbf{z}_t)/t^2$ where $\hat{\mathbf{z}}_0 = \mathbf{z}_t - tv_\theta(\mathbf{z}_t,t,c)$. Discretisation gives:

$$\pi_\theta(\mathbf{z}_{t-\Delta t} \mid \mathbf{z}_t, c) = \mathcal{N}\left(\mathbf{z}_{t-\Delta t};\ \mu_\theta^\text{flow}(\mathbf{z}_t, t, c),\ \varepsilon_t^2\Delta t I\right)$$

### Unified importance ratio

Both cases yield the same functional form:

$$\rho_t^{(i)} = \frac{\pi_\theta(\mathbf{z}_{t-1}^{(i)} \mid \mathbf{z}_t^{(i)}, c)}{\pi_{\theta_\text{old}}(\mathbf{z}_{t-1}^{(i)} \mid \mathbf{z}_t^{(i)}, c)} = \exp\left(-\frac{\Vert\mathbf{z}_{t-1}^{(i)} - \mu_\theta\Vert^2 - \Vert\mathbf{z}_{t-1}^{(i)} - \mu_{\theta_\text{old}}\Vert^2}{2\sigma_\text{SDE}^2}\right)$$

where $\sigma_\text{SDE}^2$ is $\eta_t^2 g_t^2\Delta t$ (DDPM) or $\varepsilon_t^2\Delta t$ (flow), respectively.

---

## Problem 2 — All $T$ SDE steps consume prohibitive memory for video

**Issue**: Video generation runs far more denoising steps than image generation (longer temporal sequences, more spatial tokens). Storing all $T$ SDE transitions for gradient computation is infeasible for high-resolution video.

**Idea**: Randomly subsample a $\tau$-fraction of timesteps (default $\tau = 0.6$) for the GRPO loss, rather than summing over all $T$ steps. At each iteration, draw $\mathcal{T}_\text{sub} \subset \lbrace1, \ldots, T\rbrace$ with $|\mathcal{T}_\text{sub}| = \lceil\tau T\rceil$.

**Why this works**: The GRPO gradient is an expectation over timesteps. Subsampling gives an unbiased estimate of that expectation, with variance inversely proportional to $|\mathcal{T}_\text{sub}|$. Empirically, $\tau = 0.6$ retains sufficient signal while reducing memory by $\sim 40\%$.

---

## Problem 3 — Binary and multi-modal rewards are unstable under prior methods

**Issue**: Binary rewards (0/1 safety filters, hard constraints) and multi-modal rewards (aesthetics + alignment + motion quality) destabilise training in methods that use raw reward directly (DDPO). High-variance reward signals cause large gradient updates that collapse the policy.

**Idea**: Apply group-relative advantage normalisation (GRPO standard) and multi-reward linear combination, allowing each reward component to be normalised before combination.

**Why this works**: The group baseline subtracts the mean reward, converting absolute scores to relative rankings within each prompt's group. This makes the training signal invariant to reward scale, which is especially important when mixing reward sources (e.g., aesthetic score in [0, 10] with binary safety in {0, 1}).

$$\hat{A}^{(i)} = \frac{r^{(i)} - \overline{r}}{\mathrm{std}(\lbrace{}r^{(j)}\rbrace) + \delta}, \quad r^{(i)} = \sum_k w_k r_k(x_0^{(i)}, c)$$

---

## Training Objective

PPO-clipped GRPO with timestep subsampling, applied uniformly across all backbone types:

$$\boxed{
\mathcal{L}_\text{DanceGRPO}(\theta) = -\mathbb{E}\left[\frac{1}{N_g}\sum_{i=1}^{N_g}\frac{1}{|\mathcal{T}_\text{sub}|}\sum_{t \in \mathcal{T}_\text{sub}} \min\left(\rho_t^{(i)}\hat{A}^{(i)},\ \mathrm{clip}\left(\rho_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon\right)\hat{A}^{(i)}\right)\right]
}$$

where $\mathcal{T}_\text{sub}$ is a randomly drawn $\tau$-fraction of all timesteps.

---

## Algorithm

```
Input: pretrained model θ (DDPM or flow), reward models {r_k}, prompt dist p_c,
       group size N_g, timestep fraction τ
Repeat:
  1. Sample prompts {c_j}
  2. For each c_j, generate N_g trajectories via SDE (backbone-appropriate):
       z_T^(i) ~ N(0,I)
       For t = T,...,1:
         For i = 1,...,N_g:
           DDPM:  μ_θ_old ← DDPM-SDE drift(ε_{θ_old}, z_t^(i), t, c_j)
           Flow:  μ_θ_old ← Flow-SDE drift(v_{θ_old}, z_t^(i), t, c_j)
           z_{t-1}^(i) ← μ_θ_old + σ_SDE · ε,  ε ~ N(0,I)
  3. Compute multi-reward: R^(i) = Σ_k w_k · r_k(z_0^(i), c_j)
  4. Group advantage:  Â^(i) = (R^(i) - mean) / std
  5. Subsample T_sub ← random τ-fraction of {1,...,T}
  6. For K gradient steps:
       For t ∈ T_sub, i = 1,...,N_g:
         ρ_t^(i) = π_θ(z_{t-1}^(i) | z_t^(i), c) / π_{θ_old}(...)
       L ← -mean[ min(ρ·Â, clip(ρ, 1-ε, 1+ε)·Â) ]
       θ ← θ - η ∇_θ L
  7. θ_old ← θ
```

Supported backbones: Stable Diffusion (DDPM), FLUX (flow), HunyuanVideo (flow, T2V), SkyReels-I2V (flow, I2V).

---

## Reference Implementation (VeRL-Omni)

In [`diffusion_algos.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/diffusion/diffusion_algos.py) the name `"dance_grpo"` is registered to the **same** `FlowGRPOLoss` class as `"flow_grpo"` — DanceGRPO and FlowGRPO share an identical PPO-clip objective; only the rollout (how $\mu$ and $\rho_t$ are computed, DDPM vs. flow) differs, and that lives outside the loss:

```python
@register_diffusion_loss("dance_grpo")   # same class as "flow_grpo"
def loss_dance_grpo(old_lp, lp, adv, cfg):
    c = cfg.diffusion_loss
    adv = clamp(adv, -c.adv_clip_max, c.adv_clip_max)
    ratio = exp(lp - old_lp)
    unclipped = -adv * ratio
    clipped   = -adv * clamp(ratio, 1 - c.clip_ratio, 1 + c.clip_ratio)
    return mean(max(unclipped, clipped))         # PPO-clip
```

---

## Difference from FlowGRPO

| Aspect | FlowGRPO | DanceGRPO |
|---|---|---|
| Model families | Flow matching only | DDPM + rectified flow |
| Tasks | T2I | T2I + T2V + I2V |
| Backbones | SD3.5-M, FLUX | SD, FLUX, HunyuanVideo, SkyReels |
| Timestep reduction | Fixed $T_\text{train} \ll T_\text{inf}$ | Random $\tau$-fraction subsampling |
| Efficiency variant | FlowGRPO-Fast (1-step branch) | — |
| Binary reward | Not demonstrated | Demonstrated stable |

---

## Limitations

| Problem | Addressed by |
|---|---|
| SDE noise → image artifacts → misleads reward | [CPS](cps.md) |
| Importance ratio imbalance → reward hacking | [GRPO-Guard](grpo_guard.md) |
| Full SDE still expensive; no ODE speedup | [MixGRPO](mix_grpo.md) |
| SDE blocks fast ODE samplers entirely | [DGPO](../direct_preference/dgpo.md), [AWM](../direct_preference/awm.md) |
