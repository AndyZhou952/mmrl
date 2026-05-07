# CPS — Coefficients-Preserving Sampling for RL with Flow Matching

> Notation: follows [NOTATION.md](../NOTATION.md). Flow matching: $v_\theta$, $x_t = (1-t)x_0 + t\epsilon$. CPS stochasticity parameter $\sigma_t \in [0, t-\Delta t]$.

| Field | Value |
|---|---|
| **arXiv** | [2509.05952](https://arxiv.org/abs/2509.05952) |
| **Submitted** | 2025-09-07 (revised 2025-12-08) |
| **Venue** | — (preprint) |
| **Authors** | Feng Wang, Zihao Yu |
| **Paradigm** | **Coupled** (plug-in fix; replaces SDE step, keeps GRPO objective) |
| **Cites** | FlowGRPO (2505.05470), DanceGRPO (2505.07818), DDIM (Song et al. 2020) |

---

## Motivation

FlowGRPO/DanceGRPO add noise $s_t\sqrt{\Delta t}\,\epsilon$ to the ODE step to get a stochastic policy. This noise is **uncompensated**: the resulting $x_{t-\Delta t}$ has more noise than the flow schedule specifies, pushing it off the data manifold. Reward models trained on clean images cannot reliably score these artifact-laden samples — slowing convergence. CPS fixes this by deriving the stochastic step to **preserve the linear interpolation coefficients**, keeping $x_{t-\Delta t}$ on the flow manifold.

---

## Diagnosing the artifact: coefficient mismatch

Flow matching forward process: $x_t = (1-t)x_0 + t\epsilon$. The noise level at step $t$ is $t$ (standard deviation in the noise direction).

Standard ODE-to-SDE (FlowGRPO): adds $s_t\sqrt{\Delta t}\,\epsilon_t$ to the ODE step, giving total noise standard deviation at $t-\Delta t$:
$$\sigma_\text{total} = \sqrt{(t-\Delta t)^2 + s_t^2\,\Delta t} > (t-\Delta t)$$

The excess $\sqrt{s_t^2\,\Delta t}$ is the coefficient mismatch — the sample sits **above** the schedule manifold.

---

## CPS formula

Decompose $x_{t-\Delta t}$ into three orthogonal components that exactly match the linear interpolation at $t-\Delta t$:

$$\boxed{
x_{t-\Delta t}^{\text{CPS}} = \underbrace{(1-(t-\Delta t))}_{\lambda_{t-\Delta t}}\,\hat x_0 + \underbrace{\sqrt{(t-\Delta t)^2 - \sigma_t^2}}_{\text{noise direction, on manifold}}\,\hat x_1 + \underbrace{\sigma_t}_{\text{stochasticity}}\,\epsilon_t
}$$

where:
- $\hat x_0 = x_t - t\,v_\theta(x_t,t,c)$ — Tweedie predicted clean image
- $\hat x_1 = (x_t - (1-t)\hat x_0)/t = \epsilon$ — predicted noise direction
- $\epsilon_t \sim \mathcal{N}(0,I)$ — fresh noise sample
- $\sigma_t \in [0,\; t-\Delta t]$ — tunable stochasticity (0 = deterministic ODE; $t-\Delta t$ = full re-noise)

**Coefficient preservation**: the noise-direction coefficient is set so the total noise level equals exactly $t - \Delta t$:
$$\underbrace{\sqrt{(t-\Delta t)^2 - \sigma_t^2}}_\text{old noise preserved} \oplus \underbrace{\sigma_t}_\text{new noise} \Rightarrow \sqrt{(t-\Delta t)^2 - \sigma_t^2 + \sigma_t^2} = t - \Delta t \checkmark$$

---

## Connection to DDIM

DDIM applies the same principle to DDPM. For DDPM with noise schedule $\bar\alpha_t$:
$$x_{t-1}^{\text{DDIM}} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1} - \eta^2\tilde\beta_t}\,\hat\epsilon + \eta\sqrt{\tilde\beta_t}\,\epsilon_t$$

Total noise variance: $(1-\bar\alpha_{t-1}-\eta^2\tilde\beta_t) + \eta^2\tilde\beta_t = 1-\bar\alpha_{t-1}$ ✓ (matches DDPM schedule). CPS is exactly the flow-matching analogue of stochastic DDIM.

---

## Policy density

CPS step is Gaussian — compatible with GRPO importance ratio:
$$\pi_\theta^\text{CPS}(x_{t-\Delta t} \mid x_t, c) = \mathcal{N}\!\left(x_{t-\Delta t};\; (1-(t-\Delta t))\hat x_0 + \sqrt{(t-\Delta t)^2-\sigma_t^2}\,\hat x_1,\;\; \sigma_t^2\, I\right)$$

$$\log \pi_\theta^\text{CPS} = -\frac{\|x_{t-\Delta t} - \mu_\theta^\text{CPS}(x_t,t,c)\|^2}{2\sigma_t^2} + \text{const}$$

Importance ratio (same formula as FlowGRPO, different $\mu$):
$$\rho_t^{(i)} = \frac{\pi_\theta^\text{CPS}(x_{t-\Delta t}^{(i)} \mid x_t^{(i)}, c)}{\pi_{\theta_\text{old}}^\text{CPS}(x_{t-\Delta t}^{(i)} \mid x_t^{(i)}, c)} = \exp\!\left(-\frac{\|x_{t-\Delta t}^{(i)} - \mu_\theta^\text{CPS}\|^2 - \|x_{t-\Delta t}^{(i)} - \mu_{\theta_\text{old}}^\text{CPS}\|^2}{2\sigma_t^2}\right)$$

---

## Training algorithm (drop-in replacement)

```
Replace only the SDE step in FlowGRPO / DanceGRPO / MixGRPO:

  OLD SDE step:
    x_{t-Δt} = μ_θ(x_t,t) + s_t√(Δt) ε_t       ← excess noise!

  NEW CPS step:
    x̂_0 = x_t - t · v_θ(x_t, t, c)              ← Tweedie
    x̂_1 = (x_t - (1-t)·x̂_0) / t               ← noise direction
    x_{t-Δt} = (1-(t-Δt))·x̂_0
              + √((t-Δt)² - σ_t²)·x̂_1
              + σ_t·ε_t,   ε_t ~ N(0,I)          ← manifold-preserving

GRPO objective, advantage estimation, and clipping: unchanged.
```

No change to reward calculation or update steps.

---

## Reward calculation / update

Identical to whichever base method (FlowGRPO / DanceGRPO / MixGRPO) CPS is plugged into. CPS only affects the **sampling quality** of $x_0^{(i)}$, which flows through to sharper reward signals.

---

## Limitations

- Addresses artifact problem only; does not fix ratio imbalance (→ GRPO-Guard) or SDE cost (→ MixGRPO).
- Stochasticity schedule $\{\sigma_t\}$ requires tuning.
- Assumes the flow model is well-trained; degraded base models may still produce artifacts.
