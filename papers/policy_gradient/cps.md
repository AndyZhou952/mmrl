# CPS — Coefficients-Preserving Sampling for RL with Flow Matching

> Notation: follows [NOTATION.md](../NOTATION.md). Flow matching: $x_t = (1-t)x_0 + t\epsilon$, velocity $v_\theta$. CPS stochasticity parameter: $\sigma_t \in [0, t{-}\Delta t]$; predicted noise direction: $\hat{x}_1 \approx \epsilon$.

| Field | Value |
|---|---|
| **arXiv** | [2509.05952](https://arxiv.org/abs/2509.05952) |
| **Submitted** | 2025-09-07 (revised 2025-12-08) |
| **Venue** | — (preprint) |
| **Authors** | Feng Wang, Zihao Yu |
| **GitHub** | https://github.com/IamCreateAI/FlowCPS |
| **Paradigm** | **Policy Gradient** — plug-in replacement for the SDE step; GRPO objective unchanged |
| **Cites** | FlowGRPO (2505.05470), DanceGRPO (2505.07818), DDIM (Song et al. 2020) |

---

## Context

CPS is a **drop-in fix** for FlowGRPO, DanceGRPO, and MixGRPO — it replaces only the SDE sampling step, leaving the GRPO training objective completely unchanged. Understanding requires familiarity with the ODE→SDE conversion in [FlowGRPO](flow_grpo.md). The specific problem CPS addresses is one that neither FlowGRPO-Fast, MixGRPO, nor GRPO-Guard touch: the generated images used to compute rewards are noisier than they should be, causing the reward model to evaluate off-manifold samples.

---

## Problem — ODE-to-SDE conversion injects uncompensated noise, pushing samples off the flow manifold

**Issue**: FlowGRPO converts the ODE to an SDE by adding independent Gaussian noise $s_t\sqrt{\Delta t}\epsilon_t$ at each denoising step. The rectified flow schedule specifies that $x_{t-\Delta t}$ should have noise level $t-\Delta t$ (standard deviation in the noise direction). After the SDE step, the actual noise level is:

$$\sigma_\text{actual} = \sqrt{(t-\Delta t)^2 + s_t^2\Delta t} > t-\Delta t$$

The excess $\sqrt{s_t^2\Delta t}$ is **coefficient mismatch**: the sample sits *above* the flow manifold at $t-\Delta t$. After $T$ such steps, $x_0$ is not clean — it carries residual noise that makes reward models (trained on clean images) give unreliable scores. Inaccurate reward signals corrupt the GRPO gradient.

**Idea**: Decompose the SDE step into three orthogonal components — clean image direction, noise direction, and fresh randomness — and set coefficients so the total noise level remains **exactly** $t-\Delta t$, matching the rectified flow schedule. This is the flow-matching analogue of DDIM's stochastic variant.

**Why this works**: The rectified flow interpolant $x_t = (1-t)x_0 + t\epsilon$ defines a manifold parametrised by $(t, x_0, \epsilon)$. A step that preserves the coefficients keeps $x_{t-\Delta t}$ exactly on the manifold at timestep $t-\Delta t$ — it is a valid sample from the forward process evaluated at that time, so reward models evaluate it as if it were a natural noisy image.

**Result**: CPS produces noise-free intermediate samples where Flow-SDE shows "severe noise" (Fig. 1), and the cleaner reward signal improves both final reward and convergence speed: PickScore **23.90 → 24.25** on FLUX.1-dev and **23.39 → 23.78** on FLUX.1-schnell (Tab. 2), HPSv2 **0.364 → 0.377** (Tab. 3), OCR **0.966 → 0.975** (Tab. 4); GenEval matches Flow-GRPO (both 0.97) but "converges to the optimal result at a faster speed" (Figs. 4–6).

---

## Diagnosing the Coefficient Mismatch

Standard rectified flow forward process: $x_t = (1-t)x_0 + t\epsilon$. This means:
- Clean image coefficient at time $t$: $(1-t)$
- Noise coefficient at time $t$: $t$

FlowGRPO's SDE step adds fresh noise $s_t\sqrt{\Delta t}\epsilon_t$ on top of the Euler step. After this, the noise-direction variance of $x_{t-\Delta t}$ becomes $(t-\Delta t)^2 + s_t^2\Delta t$ (due to independence of noise sources). The "schedule" noise level is just $(t-\Delta t)^2$. The mismatch grows with $s_t$ and $\Delta t$.

---

## CPS Derivation

**Goal**: construct $x_{t-\Delta t}$ with:
- Clean image component: weight $(1-(t-\Delta t))$ applied to $\hat{x}_0$
- Noise direction: weight adjusted so total noise standard deviation = $t-\Delta t$
- Fresh randomness: $\sigma_t$ (tunable; must satisfy $\sigma_t \leq t-\Delta t$)

The three-component decomposition:

$$\boxed{
x_{t-\Delta t}^\text{CPS} = \underbrace{(1-(t-\Delta t))}_{\text{clean coeff}}\hat{x}_0 + \underbrace{\sqrt{(t-\Delta t)^2 - \sigma_t^2}}_{\text{adjusted noise coeff}}\hat{x}_1 + \underbrace{\sigma_t}_{\text{fresh noise}}\epsilon_t
}$$

where:
- $\hat{x}_0 = x_t - tv_\theta(x_t,t,c)$ — Tweedie predicted clean image (same as FlowGRPO)
- $\hat{x}_1 = (x_t - (1-t)\hat{x}_0)/t$ — estimated noise direction ($\approx \epsilon$ from forward process)
- $\epsilon_t \sim \mathcal{N}(0,I)$ — fresh randomness
- $\sigma_t \in [0, t-\Delta t]$ — stochasticity level ($\sigma_t = 0$: deterministic DDIM-style; $\sigma_t = t-\Delta t$: maximum stochasticity)

**Coefficient preservation check**: the noise-direction variance of $x_{t-\Delta t}^\text{CPS}$ is $(t-\Delta t)^2 - \sigma_t^2 + \sigma_t^2 = (t-\Delta t)^2$, so the noise standard deviation is exactly $t-\Delta t$ ✓.

### Connection to stochastic DDIM

DDIM (Song et al. 2020) applies the identical principle to DDPM:

$$x_{t-1}^\text{DDIM} = \sqrt{\bar\alpha_{t-1}}\hat{x}_0 + \sqrt{1-\bar\alpha_{t-1} - \eta^2\tilde\beta_t}\hat\epsilon + \eta\sqrt{\tilde\beta_t}\epsilon_t$$

The middle term is set so that total noise variance equals $1-\bar\alpha_{t-1}$ (matching the DDPM schedule). CPS is exactly the rectified-flow analogue of stochastic DDIM.

---

## Policy Density

The CPS step is still Gaussian, so the GRPO importance ratio formula is unchanged — only $\mu_\theta$ and $\sigma^2$ differ:

$$\pi_\theta^\text{CPS}(x_{t-\Delta t} \mid x_t, c) = \mathcal{N}\left(x_{t-\Delta t};\ (1-(t-\Delta t))\hat{x}_0 + \sqrt{(t-\Delta t)^2-\sigma_t^2}\hat{x}_1,\ \sigma_t^2 I\right)$$

$$\rho_t^{(i)} = \exp\left(-\frac{\Vert{}x_{t-\Delta t}^{(i)} - \mu_\theta^\text{CPS}\Vert^2 - \Vert{}x_{t-\Delta t}^{(i)} - \mu_{\theta_\text{old}}^\text{CPS}\Vert^2}{2\sigma_t^2}\right)$$

---

## Algorithm (Drop-In Replacement)

```
Replace only the SDE sampling step in FlowGRPO / DanceGRPO / MixGRPO:

  OLD (FlowGRPO SDE step) — coefficient mismatch:
    x̂_0     ← x_t - t · v_{θ_old}(x_t, t, c)
    μ_{θ_old} ← x_t - v_{θ_old}·Δt + (s_t²Δt/2t²)·(x̂_0 - x_t)
    x_{t-Δt} ← μ_{θ_old} + s_t√Δt · ε_t           ← excess noise!

  NEW (CPS step) — coefficients preserved:
    x̂_0     ← x_t - t · v_{θ_old}(x_t, t, c)       # Tweedie
    x̂_1     ← (x_t - (1-t)·x̂_0) / t               # noise direction
    x_{t-Δt} ← (1-(t-Δt))·x̂_0
              + √((t-Δt)² - σ_t²)·x̂_1
              + σ_t·ε_t,   ε_t ~ N(0,I)              # manifold-preserving

  CPS importance ratio (replace μ in the ratio formula):
    μ_θ^CPS ← (1-(t-Δt))·x̂_0(v_θ) + √((t-Δt)²-σ_t²)·x̂_1(v_θ)
    μ_{θ_old}^CPS ← same formula with v_{θ_old}
    ρ_t^(i) ← exp(-(‖x_{t-Δt}^(i) - μ_θ^CPS‖² - ‖x_{t-Δt}^(i) - μ_{θ_old}^CPS‖²) / (2σ_t²))

GRPO advantage, clip objective, KL penalty: all unchanged.
```

Reward calculation and gradient update steps are identical to the base method.

---

## Reference Implementation (VeRL-Omni)

CPS is **not a loss** — it is a sampler variant selected by the config flag `sde_type: sde → cps` (FlowGRPO/DanceGRPO/MixGRPO keep their loss). It lives in [`flow_match_sde.py::sample_previous_step`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/pipelines/schedulers/flow_match_sde.py) (enabled e.g. in `examples/flowgrpo_trainer/run_sd35_medium_ocr_lora.sh` via `rollout.algo.sde_type="cps"`). Here `sigma` is the flow noise level at the current step and `sigma_prev` at the next; the only difference from the default `"sde"` branch is how `prev_sample_mean` and the noise term are built:

```python
# sde_type == "sde"  (FlowGRPO): Euler step + extra sqrt(-dt) noise → noise level overshoots schedule
std_dev_t        = sqrt(sigma / (1 - sigma)) * noise_level
prev_sample_mean = sample * (1 + std_dev_t**2/(2*sigma) * dt) \
                 + model_output * (1 + std_dev_t**2*(1-sigma)/(2*sigma)) * dt
prev_sample      = prev_sample_mean + std_dev_t * sqrt(-dt) * noise          # <<< uncompensated noise

# sde_type == "cps": decompose into clean + noise directions, coefficients preserve the schedule
std_dev_t        = sigma_prev * sin(noise_level * pi/2)                       # <<< σ_t (bounded by σ_prev)
x0_hat           = sample - sigma * model_output                             # <<< Tweedie clean image
eps_hat          = sample + model_output * (1 - sigma)                       # <<< noise direction
prev_sample_mean = x0_hat * (1 - sigma_prev) + eps_hat * sqrt(sigma_prev**2 - std_dev_t**2)  # <<< coeff-preserving
prev_sample      = prev_sample_mean + std_dev_t * noise                      # <<< no sqrt(-dt): level == σ_prev
```

The `cps` `prev_sample_mean` is exactly the boxed CPS formula with $t{-}\Delta t \leftrightarrow$ `sigma_prev`, $\sigma_t \leftrightarrow$ `std_dev_t`; the variance check $(\sigma_\text{prev}^2 - \sigma_t^2) + \sigma_t^2 = \sigma_\text{prev}^2$ is what keeps the sample on-manifold. (`dance_sde` is a third, numerically-stable variant in the same function.)

---

## Stochasticity Schedule

$\sigma_t$ is a hyperparameter per timestep with range $[0, t-\Delta t]$:

| Setting | Effect |
|---|---|
| $\sigma_t = 0$ | Deterministic (DDIM-style); no gradient signal from stochasticity |
| $\sigma_t = t-\Delta t$ | Maximum stochasticity; fully re-sampled noise direction |
| $\sigma_t$ moderate | Balances exploration (reward diversity) vs. manifold fidelity |

The paper finds a moderate cosine schedule (peaking at intermediate $t$) works best.

---

## Limitations

- Addresses the artifact problem only — does not reduce SDE compute cost (→ [MixGRPO](mix_grpo.md)) or ratio imbalance (→ [GRPO-Guard](grpo_guard.md)).
- The stochasticity schedule $\lbrace\sigma_t\rbrace$ requires tuning per model and reward type.
- Assumes the base flow model is well-trained; a poor $v_\theta$ produces an unreliable $\hat{x}_0$ and $\hat{x}_1$, degrading manifold preservation.
