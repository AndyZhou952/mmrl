# GRPO-Guard — Mitigating Implicit Over-Optimization via Regulated Clipping

> Notation: follows [NOTATION.md](../NOTATION.md). $\rho_t^{(i)}$ = importance ratio at step $t$ for sample $i$; $\hat\rho_t^{(i)}$ = normalised ratio (after RatioNorm). Gradient reweighting factor $\delta_t$.

| Field | Value |
|---|---|
| **arXiv** | [2510.22319](https://arxiv.org/abs/2510.22319) |
| **Submitted** | 2025-10-25 (revised 2025-10-30) |
| **Venue** | — (preprint) |
| **Authors** | Jing Wang, Jiajun Liang, Jie Liu, Henglin Liu, Gongye Liu, Jun Zheng, Wanyuan Pang, Ao Ma, Zhenyu Xie, Xintao Wang, Meng Wang, Pengfei Wan, Xiaodan Liang |
| **Paradigm** | **Coupled** (modifies the ratio computation inside the GRPO objective) |
| **Cites** | FlowGRPO (2505.05470), DanceGRPO (2505.07818), PPO, GRPO (DeepSeek-R1) |

---

## Motivation

PPO clipping assumes the importance ratio $\rho_t$ is centred near 1 and has similar variance across update steps. In flow-matching GRPO, empirical analysis shows:
1. **Mean of $\rho_t$ is systematically below 1** — the new policy overshoots the old policy's mean, so the ratio distribution is left-shifted.
2. **Variance of $\rho_t$ differs substantially across timesteps** — early denoising steps (high noise level $t$) have small variance; late steps (low $t$) have large variance.

Consequence: the clip band $[1-\epsilon, 1+\epsilon]$ is effectively inactive for most positive-advantage samples (since $\rho_t < 1-\epsilon$ often), allowing unconstrained large updates on high-reward samples. This drives **reward hacking**: proxy reward improves but true image quality degrades.

---

## Diagnosing the failure

For a Gaussian policy $\pi_\theta(x_{t-\Delta t} \mid x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2\Delta t\, I)$, the log-ratio is:
$$\log \rho_t^{(i)} = \frac{\|x_{t-\Delta t}^{(i)} - \mu_{\theta_\text{old}}\|^2 - \|x_{t-\Delta t}^{(i)} - \mu_\theta\|^2}{2\,\sigma_t^2\,\Delta t} = \frac{(\mu_\theta - \mu_{\theta_\text{old}}) \cdot \Delta\mu_t}{{\sigma_t^2\,\Delta t}}$$

where $\Delta\mu_t = \mu_\theta - \mu_{\theta_\text{old}}$. The magnitude depends on $\sigma_t$ — steps with small $\sigma_t$ (late denoising) produce large log-ratios, while steps with large $\sigma_t$ (early denoising) produce small ones. This imbalance is the root cause.

---

## Key contribution 1 — RatioNorm (per-timestep standardisation)

Replace the raw log importance ratio with a **standardised** version that has zero mean and unit standard deviation at each timestep $t$:

$$\log \hat\rho_t^{(i)} = \frac{\log \rho_t^{(i)} - \mathbb{E}_i[\log \rho_t^{(i)}]}{\text{std}_i(\log \rho_t^{(i)}) + \delta} + 0$$

Equivalently, in the paper's formulation using the Gaussian structure:
$$\log \hat\rho_t = \sigma_t\sqrt{\Delta t}\!\left(\log \rho_t + \frac{\|\Delta\mu_t\|^2}{2\sigma_t^2\Delta t}\right) = -\Delta\mu_t \cdot \epsilon_t$$

where $\epsilon_t$ is the noise used at that step. This form removes the timestep-dependent scale factor $1/(\sigma_t^2\Delta t)$ and replaces it with a unit-scale projection onto the noise direction.

After RatioNorm: $\mathbb{E}[\log \hat\rho_t] \approx 0$ and $\text{std}(\log \hat\rho_t) \approx 1$ for all $t$ — matching PPO's design assumption.

The normalised ratio is then:
$$\hat\rho_t^{(i)} = \exp(\log \hat\rho_t^{(i)})$$

---

## Key contribution 2 — Gradient reweighting

Even after RatioNorm, gradient magnitudes vary across timesteps due to differences in how $\mu_\theta$ depends on $v_\theta$ at each $t$. Introduce a per-timestep reweighting factor $\delta_t$:

$$\delta_t = \begin{cases} 1/\Delta t & \text{(Flow-GRPO)} \\ \beta_t/\Delta t & \text{(DanceGRPO, } \beta_t \text{ from DDPM schedule)} \end{cases}$$

This normalises gradient contributions so that each timestep contributes roughly equally to the parameter update, reducing the empirically observed ~20× variation to ~2.5×.

---

## Training objective

$$\boxed{
\mathcal{L}_\text{GRPO-Guard}(\theta) = -\mathbb{E}\!\left[\frac{1}{N_g}\sum_{i=1}^{N_g} \frac{1}{T}\sum_{t=1}^T \delta_t \cdot \min\!\left(\hat\rho_t^{(i)}\hat A^{(i)},\;\text{clip}\!\left(\hat\rho_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon\right)\hat A^{(i)}\right)\right]
}$$

Compared to plain FlowGRPO/DanceGRPO:
- $\rho_t$ → $\hat\rho_t$ (RatioNorm applied)
- Extra $\delta_t$ factor (gradient reweighting)
- Everything else (advantage, clip, KL) unchanged

---

## Sampling and reward calculation

Identical to whichever base method (FlowGRPO or DanceGRPO) GRPO-Guard is applied to.

---

## Training algorithm

```
Replace the GRPO objective computation:

  OLD:
    ρ_t^(i) = π_θ(x_{t-1}^(i)|x_t^(i)) / π_{θ_old}(...)
    loss += min(ρ_t^(i)·Â^(i), clip(ρ_t^(i),1-ε,1+ε)·Â^(i))

  NEW (GRPO-Guard):
    log_ρ_raw = (||x_{t-1}^(i) - μ_θ_old||² - ||x_{t-1}^(i) - μ_θ||²) / (2σ_t²Δt)
    # RatioNorm: standardise across group members at this timestep
    log_ρ_hat = σ_t√(Δt) · (log_ρ_raw + ||Δμ_t||² / (2σ_t²Δt))
              = -Δμ_t · ε_t                          ← unit-scale
    ρ_hat = exp(log_ρ_hat)
    # Gradient reweighting
    δ_t = 1/Δt    (or β_t/Δt for DDPM)
    loss += δ_t · min(ρ_hat·Â^(i), clip(ρ_hat,1-ε,1+ε)·Â^(i))

All other parts of the training loop unchanged.
```

---

## Effect

| Metric | FlowGRPO (no Guard) | GRPO-Guard |
|---|---|---|
| Ratio mean | $< 1$ (left-shifted) | $\approx 1$ |
| Ratio variance across timesteps | ~20× variation | ~2.5× variation |
| Reward hacking | Occurs after prolonged training | Mitigated |
| Proxy reward | Eventually diverges (reward hacking) | Stable improvement |

---

## Compatibility

GRPO-Guard is a **modifier** — it can be applied to any coupled GRPO method:
- FlowGRPO + Guard ✓
- DanceGRPO + Guard ✓
- MixGRPO + Guard ✓ (apply within window only)
- CPS + Guard ✓ (use CPS for sampling, Guard for objective)

---

## Limitations

- RatioNorm slightly approximates the exact normalisation (uses per-group statistics, not population).
- Gradient reweighting $\delta_t$ is a fixed schedule, not adaptive.
- Does not address the SDE noise artifact problem (→ CPS) or computational expense (→ MixGRPO).
