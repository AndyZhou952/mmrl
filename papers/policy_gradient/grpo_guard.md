# GRPO-Guard — Mitigating Implicit Over-Optimization via Regulated Clipping

> Notation: follows [NOTATION.md](../NOTATION.md). Raw importance ratio: $\rho_t^{(i)}$; RatioNorm-normalised ratio: $\hat\rho_t^{(i)}$; policy mean shift: $\Delta\mu_t = \mu_\theta - \mu_{\theta_\text{old}}$; gradient reweighting factor: $\delta_t$.

| Field | Value |
|---|---|
| **arXiv** | [2510.22319](https://arxiv.org/abs/2510.22319) |
| **Submitted** | 2025-10-25 (revised 2025-10-30) |
| **Venue** | — (preprint) |
| **Authors** | Jing Wang, Jiajun Liang, Jie Liu, Henglin Liu, Gongye Liu, Jun Zheng, Wanyuan Pang, Ao Ma, Zhenyu Xie, Xintao Wang, Meng Wang, Pengfei Wan, Xiaodan Liang |
| **Paradigm** | **Policy Gradient** — modifier applied inside the GRPO objective; no change to sampling |
| **Cites** | FlowGRPO (2505.05470), DanceGRPO (2505.07818), PPO, GRPO (DeepSeek-R1) |

---

## Context

GRPO-Guard is a **modifier** applied to the training objective of any policy-gradient GRPO method (FlowGRPO, DanceGRPO, MixGRPO). It changes nothing about how images are sampled; it only changes how the importance ratio is computed and weighted inside the loss. The core issue it addresses is that PPO's clip mechanism — designed around the assumption $\mathbb{E}[\rho_t] \approx 1$ — breaks down in flow-matching GRPO, causing reward hacking. This file builds on the GRPO objective from [flow_grpo.md](flow_grpo.md).

---

## Problem 1 — The importance ratio is systematically left-shifted; PPO clipping does not engage

**Issue**: In flow-matching GRPO, empirical measurement shows that the mean of $\rho_t^{(i)}$ is **systematically below 1** across all timesteps. PPO clipping is designed to catch ratios outside $[1-\epsilon, 1+\epsilon]$, assuming the ratio is centred near 1. When the mean is left-shifted (e.g., mean $\approx 0.7$), most positive-advantage samples already satisfy $\rho_t^{(i)} < 1-\epsilon$, so the clip is never activated — the update is unconstrained. This allows the policy to take unbounded steps on high-reward samples, driving **reward hacking**: proxy reward keeps climbing but image quality degrades.

**Why it happens**: For a Gaussian policy at step $t$, the log importance ratio is:

$$\log \rho_t^{(i)} = \frac{(\mu_\theta - \mu_{\theta_\text{old}}) \cdot (x_{t-\Delta t}^{(i)} - \mu_{\theta_\text{old}})}{\sigma_t^2\Delta t}$$

When $\theta$ has been updated in a direction that increases reward, $\mu_\theta$ moves away from $\mu_{\theta_\text{old}}$, and the numerator tends to be negative for randomly-drawn $x_{t-\Delta t}^{(i)}$ (the sample is "left behind" the shifting mean). This structurally left-shifts $\rho_t$.

**Idea — RatioNorm**: Standardise the log importance ratio within the group at each timestep, removing the systematic bias and scale. Define the normalised log ratio:

$$\log \hat\rho_t^{(i)} = \frac{\log \rho_t^{(i)} - \mathbb{E}_i[\log \rho_t^{(i)}]}{\mathrm{std}_i(\log \rho_t^{(i)}) + \delta}$$

Using the Gaussian structure of the flow policy, this simplifies to a form involving only the dot product of the mean shift with the noise sample:

$$\log \hat\rho_t^{(i)} = -\Delta\mu_t \cdot \epsilon_t^{(i)}$$

where $\epsilon_t^{(i)}$ is the noise injected at step $t$ for sample $i$.

**Why this works**: After RatioNorm, $\mathbb{E}[\log\hat\rho_t] \approx 0$ and $\mathrm{std}(\log\hat\rho_t) \approx 1$ for all $t$. This matches PPO's design assumption — the clip band $[1-\epsilon, 1+\epsilon]$ now activates as intended, constraining large updates when the policy has drifted far from the old policy.

**Result**: RatioNorm restores a balanced, step-consistent ratio (Fig. 2) so the clip engages — the payoff shows in the **"Gold score"** (true quality, vs proxy reward): on SD3.5-M it rises while the proxy holds — GenEval Gold **0.84 → 0.89**, TextRender Gold **0.88 → 0.99**, PickScore Gold **1.16 → 1.20**; on FLUX.1-dev GenEval Gold **0.88 → 1.02** (Tab. 1) — i.e. much less over-optimization at equal-or-better proxy reward.

---

## Problem 2 — Gradient magnitude varies ~20× across timesteps; late steps dominate

**Issue**: Even after RatioNorm, the gradient magnitude contributed by each timestep is not uniform. For a Gaussian policy, the gradient of $\log\rho_t$ with respect to $\theta$ scales as $1/(\sigma_t^2\Delta t)$. Steps with small $\sigma_t$ (late denoising, low noise) produce large gradients; steps with large $\sigma_t$ (early denoising, high noise) produce small ones. The ~20× variation means late timesteps dominate training, causing the model to over-optimise the final appearance while under-optimising the coarse structure.

**Idea — Gradient reweighting**: Multiply each timestep's loss contribution by a per-timestep factor $\delta_t$ that equalises gradient magnitudes:

$$\delta_t = \begin{cases} 1/\Delta t & \text{for flow matching (FlowGRPO, MixGRPO)} \\ \beta_t/\Delta t & \text{for DDPM (DanceGRPO), where } \beta_t \text{ is the noise schedule} \end{cases}$$

**Why this works**: $\delta_t$ scales each step's contribution inversely proportional to its variance, restoring approximately uniform gradient magnitudes across timesteps.

**Result**: Gradient-magnitude variation across timesteps drops from **~20× to ~2.5×** (Fig. 3), so coarse-structure (high-noise) steps are no longer drowned out by fine-detail steps — this is what makes the Gold-score gains in Problem 1 stable rather than transient.

---

## Training Objective

$$\boxed{
\mathcal{L}_\text{GRPO-Guard}(\theta) = -\mathbb{E}\left[\frac{1}{N_g}\sum_{i=1}^{N_g}\frac{1}{T}\sum_{t=1}^{T}\delta_t\cdot\min\left(\hat\rho_t^{(i)}\hat{A}^{(i)},\ \mathrm{clip}\left(\hat\rho_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon\right)\hat{A}^{(i)}\right)\right]
}$$

Compared to plain FlowGRPO: $\rho_t \to \hat\rho_t$ (RatioNorm) and an extra $\delta_t$ factor. Sampling, advantage estimation, and KL penalty are unchanged.

---

## Algorithm (Objective Modifier)

```
Replace only the GRPO objective computation in FlowGRPO / DanceGRPO / MixGRPO:

  OLD:
    log_ρ_t^(i) = (‖x_{t-Δt}^(i) - μ_{θ_old}‖² - ‖x_{t-Δt}^(i) - μ_θ‖²) / (2σ_t²Δt)
    ρ_t^(i)     = exp(log_ρ_t^(i))
    loss        += min(ρ_t^(i)·Â^(i),  clip(ρ_t^(i), 1-ε, 1+ε)·Â^(i))

  NEW (GRPO-Guard):
    log_ρ_raw = (‖x_{t-Δt}^(i) - μ_{θ_old}‖² - ‖x_{t-Δt}^(i) - μ_θ‖²) / (2σ_t²Δt)

    # RatioNorm: project onto noise direction → unit scale, zero mean
    Δμ_t      = μ_θ(x_t^(i), t, c) - μ_{θ_old}(x_t^(i), t, c)
    log_ρ_hat = -Δμ_t · ε_t^(i)                           ← ε_t^(i) is stored SDE noise
    ρ_hat     = exp(log_ρ_hat)

    # Gradient reweighting
    δ_t       = 1/Δt        (flow)   or   β_t/Δt  (DDPM)
    loss      += δ_t · min(ρ_hat·Â^(i),  clip(ρ_hat, 1-ε, 1+ε)·Â^(i))

All sampling steps and advantage computation unchanged.
```

---

## Reference Implementation (VeRL-Omni)

Condensed from [`GRPOGuardLoss` in `diffusion_algos.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/diffusion/diffusion_algos.py) (`@register_diffusion_loss("grpo_guard")`). It is **identical to `FlowGRPOLoss` except the four `# <<<` lines** — it adds a reverse-SDE mean-drift bias projected onto the per-step scale `sqrt_dt·σ_t` (RatioNorm), then rescales the loss by `1/sqrt_dt²` so gradient magnitude is consistent across timesteps ($\delta_t = 1/\Delta t$). The extra inputs `prev_sample_mean`, `old_prev_sample_mean`, `std_dev_t`, `sqrt_dt` come from the rollout:

```python
@register_diffusion_loss("grpo_guard")
def loss_grpo_guard(old_lp, lp, adv, mean_θ, mean_old, std_t, sqrt_dt, cfg):
    c = cfg.diffusion_loss
    adv = clamp(adv, -c.adv_clip_max, c.adv_clip_max)              # same as FlowGRPO
    scale        = sqrt_dt.mean() * std_t.mean()                   # <<< shared per-step scalar
    mean_diff_sq = ((mean_θ - mean_old) ** 2).mean(non_batch_dims) # <<< reverse-SDE mean drift
    ratio_bias   = mean_diff_sq / (2 * scale ** 2)                 # <<< RatioNorm bias
    ratio = exp((lp - old_lp + ratio_bias) * scale)               # <<< FlowGRPO: exp(lp - old_lp)
    unclipped = -adv * ratio                                      # ┐
    clipped   = -adv * clamp(ratio, 1 - c.clip_ratio, 1 + c.clip_ratio)  # ├ identical PPO-clip body
    return mean(max(unclipped, clipped)) / sqrt_dt.mean() ** 2     # <<< FlowGRPO: mean(...); here ÷Δt
```

---

## Effect on Training Stability

| Metric | FlowGRPO baseline | With GRPO-Guard |
|---|---|---|
| Mean of $\rho_t$ | $<1$ (left-shifted) | $\approx 1$ (centred) |
| Variance spread across timesteps | $\sim 20\times$ | $\sim 2.5\times$ |
| PPO clip activation | Rarely | As designed |
| Reward hacking onset | Early (reward climbs, quality drops) | Substantially delayed |

---

## Compatibility

GRPO-Guard is a pure objective modifier — it is composable with all policy-gradient methods:
- FlowGRPO + Guard ✓
- DanceGRPO + Guard ✓
- MixGRPO + Guard ✓ (apply RatioNorm and $\delta_t$ to window steps only)
- CPS + Guard ✓ (CPS handles sampling quality; Guard handles objective stability)
- UniGRPO adopts RatioNorm from this work ✓

---

## Limitations

- RatioNorm uses per-group statistics (group size $N_g$) rather than population statistics — the normalisation is an approximation that degrades for small groups ($N_g < 4$).
- Gradient reweighting $\delta_t$ is a fixed schedule, not adaptive to the current model's gradient landscape.
- Does not address SDE noise artifacts (→ [CPS](cps.md)) or computational cost (→ [MixGRPO](mix_grpo.md)).
