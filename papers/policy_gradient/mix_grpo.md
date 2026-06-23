# MixGRPO — Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE

> Notation: follows [NOTATION.md](../NOTATION.md). Flow matching: $v_\theta$, $t \in [0,1]$. Window $\mathcal{W}(l)$ is a contiguous block of denoising steps; SDE diffusion coefficient inside window: $\sigma_t$.

| Field | Value |
|---|---|
| **arXiv** | [2507.21802](https://arxiv.org/abs/2507.21802) |
| **Submitted** | 2025-07-29 (revised 2026-03-20) |
| **Venue** | — (preprint) |
| **Authors** | Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, Liefeng Bo |
| **GitHub** | https://github.com/Tencent-Hunyuan/MixGRPO |
| **Paradigm** | **Policy Gradient** — SDE + gradient confined to a sliding window; ODE everywhere else |
| **Cites** | FlowGRPO (2505.05470), DanceGRPO (2505.07818), DDPO, GRPO |
| **Cited by** | HunyuanImage 3.0 (production training pipeline) |

---

## Context

FlowGRPO and DanceGRPO established that applying GRPO to flow matching requires converting the ODE to an SDE for every denoising step — giving a tractable importance ratio but forcing the entire $T$-step trajectory through SDE sampling. MixGRPO is the first work to ask: **do all $T$ steps actually need to be stochastic?** The answer is no: only the steps that contribute to the importance ratio and gradient need SDE treatment. This file builds on the FlowGRPO background; see [flow_grpo.md](flow_grpo.md) for the ODE→SDE derivation.

---

## Problem 1 — Full $T$-step SDE is slow and blocks high-quality ODE solvers

**Issue**: FlowGRPO and DanceGRPO run SDE sampling for all $T$ denoising steps. Two compounding costs arise: (1) SDE steps are first-order Euler-Maruyama — they cannot use high-order ODE solvers (DPM-Solver++, Heun) that achieve the same quality with $2\text{–}3\times$ fewer steps; (2) gradient tracking through all $T$ SDE steps requires storing all $T$ intermediate activations — $O(T \cdot d)$ peak memory.

**Idea**: Confine both SDE sampling and gradient tracking to a **contiguous sliding window** $\mathcal{W}(l) = \lbrace{}l, l{+}1, \ldots, l{+}w{-}1\rbrace$ of $w$ denoising steps. Outside the window, use a standard deterministic ODE with no gradient tracking. Slide the window through the trajectory over training iterations, so different segments accumulate gradient updates over time.

**Why this works**: The GRPO importance ratio $\rho_t$ is meaningful only at SDE steps (ODE steps have $\rho_t = 1$ trivially, contributing nothing to the gradient). Confining SDE to $w$ steps reduces memory to $O(w \cdot d)$, and allows the non-window steps to use fast ODE solvers — improving trajectory quality at the tail (high detail) and head (coarse structure). Sliding the window ensures the entire trajectory is eventually covered across iterations, so the full reward landscape is optimised even though only $w$ steps are updated at each iteration.

**Result**: MixGRPO achieves **nearly 50% lower training time than DanceGRPO** while *outperforming* it across multiple dimensions of human-preference alignment (abstract) — i.e. the sliding window removes most of the SDE/gradient cost without sacrificing reward.

### Three trajectory zones

For a trajectory with $T$ steps, window starting at position $l$:

$$\underbrace{x_T \to \cdots \to x_{l+w}}_{\text{ODE, no gradient}} \Bigg| \underbrace{x_{l+w} \to \cdots \to x_l}_{\text{SDE + gradient, window } \mathcal{W}(l)} \Bigg| \underbrace{x_l \to \cdots \to x_0}_{\text{ODE, no gradient}}$$

**Outside window** (ODE, no grad):

$$x_{t-\Delta t} = x_t - v_\theta(x_t, t, c)\Delta t$$

**Inside window** (SDE with score correction, with grad):

$$x_{t-\Delta t} = x_t - v_\theta(x_t,t,c)\Delta t + \frac{\sigma_t^2\Delta t}{2t^2}(\hat{x}_0 - x_t) + \sigma_t\sqrt{\Delta t}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,I)$$

where $\hat{x}_0 = x_t - tv_\theta(x_t,t,c)$ is the Tweedie estimate. The per-step transition inside the window is Gaussian:

$$\pi_\theta(x_{t-\Delta t} \mid x_t, c) = \mathcal{N}\left(x_{t-\Delta t};\ \mu_\theta(x_t,t,c),\ \sigma_t^2\Delta t I\right), \quad t \in \mathcal{W}(l)$$

**Window sliding** (every $\tau$ training iterations, stride $s$):

$$l \leftarrow \min(l + s,\ T - w)$$

---

## Problem 2 — Window width is fixed; extreme compression still needs high-order ODE

**Issue**: Even with a window of $w=4$ steps out of $T=25$ (the paper's optimum), the 21 ODE steps outside the window still use first-order Euler. This is acceptable for moderate $T$, but for larger $T$ or video (where $T$ may reach 100+), Euler ODE quality degrades. A variant that uses a high-order ODE solver outside the window would compound the speedup.

**Idea — MixGRPO-Flash**: Shrink the window to $w=1$ step and replace all ODE steps with **DPM-Solver++** at compression ratio $\tilde{r} < 1$:

$$\tilde{T} = 1 + (T-1)\tilde{r} \qquad \text{(effective total steps after solver compression)}$$

**Why this works**: DPM-Solver++ is a high-order (second/third order) solver that achieves Euler-equivalent quality at $\sim 2\text{–}3\times$ fewer steps. Combining $w=1$ with $\tilde{r} \approx 0.3\text{–}0.4$ gives a total speedup:

$$S = T/\tilde{T} \approx 1/\tilde{r}$$

**Result**: MixGRPO-Flash **further reduces training time by 71%** (abstract) with almost no degradation in reward performance — the largest efficiency gain in the policy-gradient chain.

---

## Training Objective

GRPO-clip loss restricted to window steps only; ODE steps contribute nothing:

$$\boxed{
\mathcal{L}_\text{MixGRPO}(\theta) = -\mathbb{E}\left[\frac{1}{N_g}\sum_{i=1}^{N_g}\frac{1}{|\mathcal{W}|}\sum_{t \in \mathcal{W}(l)} \min\left(\rho_t^{(i)}\hat{A}^{(i)},\ \mathrm{clip}\left(\rho_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon\right)\hat{A}^{(i)}\right)\right]
}$$

Importance ratio (window steps only):

$$\rho_t^{(i)} = \exp\left(-\frac{\Vert{}x_{t-\Delta t}^{(i)} - \mu_\theta\Vert^2 - \Vert{}x_{t-\Delta t}^{(i)} - \mu_{\theta_\text{old}}\Vert^2}{2\sigma_t^2\Delta t}\right), \quad t \in \mathcal{W}(l)$$

---

## Algorithm

```
Input: pretrained v_θ, reward r, prompt dist p_c, N_g,
       window width w, slide interval τ_iter, stride s, total steps T
Initialize: l ← 0
Repeat:
  Every τ_iter iterations:  l ← min(l + s, T - w)
  1. Sample prompts {c_j}
  2. For each c_j, generate N_g trajectories (mixed ODE/SDE):
       x_T^(i) ~ N(0,I)
       ODE (no grad):  x_T^(i) → x_{l+w}^(i)          # steps T ... l+w+1
       SDE (with grad): x_{l+w}^(i) → x_l^(i)          # steps l+w ... l+1 (window)
         For t = l+w, ..., l+1:
           x̂_0 ← x_t - t·v_{θ_old}(x_t^(i), t, c)
           μ_{θ_old} ← x_t - v_{θ_old}·Δt + (σ_t²Δt/2t²)·(x̂_0 - x_t)
           x_{t-Δt}^(i) ← μ_{θ_old} + σ_t√Δt·ε,  ε ~ N(0,I)
       ODE (no grad):  x_l^(i) → x_0^(i)               # steps l ... 1
  3. Compute R^(i) = r(x_0^(i), c_j)
  4. Â^(i) = (R^(i) - mean) / std
  5. For K gradient steps:
       For t ∈ W(l), i = 1,...,N_g:
         μ_θ ← x_t - v_θ·Δt + (σ_t²Δt/2t²)·(x̂_0 - x_t)   # current θ
         ρ_t^(i) = exp(-(‖x_{t-Δt}^(i) - μ_θ‖² - ‖x_{t-Δt}^(i) - μ_{θ_old}‖²) / (2σ_t²Δt))
         L ← -mean[ min(ρ·Â, clip(ρ, 1-ε, 1+ε)·Â) ]
         θ ← θ - η ∇_θ L
  6. θ_old ← θ

MixGRPO-Flash variant: replace ODE steps outside window with DPM-Solver++ at rate r̃;
  shrink window to w=1; total effective steps T̃ = 1 + (T-1)·r̃.
```

---

## Efficiency Comparison

| Method | SDE steps / traj | Grad steps | ODE solver outside | Reported speedup |
|---|---|---|---|---|
| FlowGRPO | All $T$ | All $T$ | Euler (N/A) | 1× (baseline) |
| FlowGRPO-Fast | 1–2 (branch point) | 1–2 | Euler | ~$T/2$× |
| MixGRPO | $w$ (sliding) | $w$ | Euler | $T/w$× |
| MixGRPO-Flash | 1 | 1 | DPM-Solver++ | **71%** time reduction |

Recommended hyperparameters (ablated in paper): $w=4$, $\tau_\text{iter}=25$, $s=1$ for $T=25$.

---

## Limitations

| Problem | Note |
|---|---|
| Window width $w$ is a hyperparameter | Sensitivity studied in paper; $w=4$ recommended |
| SDE noise inside window still causes artifacts | [CPS](cps.md) can replace the SDE step inside $\mathcal{W}(l)$ |
| Ratio imbalance within window persists | [GRPO-Guard](grpo_guard.md) is compatible |
| Still requires SDE for any gradient | Fully solver-agnostic: see [DGPO](../direct_preference/dgpo.md), [AWM](../direct_preference/awm.md) |
