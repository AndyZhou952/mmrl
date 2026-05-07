# MixGRPO — Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE

> Notation: follows [NOTATION.md](../NOTATION.md). Flow matching convention: $v_\theta$, $t \in [0,1]$. Window indices in discrete steps. SDE diffusion coefficient: $\sigma_t$.

| Field | Value |
|---|---|
| **arXiv** | [2507.21802](https://arxiv.org/abs/2507.21802) |
| **Submitted** | 2025-07-29 (revised 2026-03-20) |
| **Venue** | — (preprint) |
| **Authors** | Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, Liefeng Bo |
| **GitHub** | https://github.com/Tencent-Hunyuan/MixGRPO |
| **Paradigm** | **Coupled** — per-step log-prob within window; ODE outside |
| **Cites** | FlowGRPO (2505.05470), DanceGRPO (2505.07818), DDPO, GRPO |
| **Cited by** | HunyuanImage 3.0 (uses MixGRPO in production training) |

---

## Motivation

FlowGRPO and DanceGRPO run SDE for all $T$ denoising steps. Two costs arise: (1) SDE cannot use high-order solvers; (2) gradient tracking through all $T$ steps is memory-intensive. **Key insight**: only SDE steps contribute to the importance ratio and gradient. MixGRPO confines SDE and gradient to a contiguous **sliding window** of $w$ steps, using fast ODE elsewhere.

---

## Setting

Same as FlowGRPO/DanceGRPO: flow matching, $N_g$ images per prompt, group-relative advantage.

**Window** $\mathcal{W}(l) = \{l, l{+}1, \ldots, l{+}w{-}1\}$: contiguous block of $w$ denoising steps starting at position $l$.
**Hyperparameters**: $w$ (width), $\tau$ (shift interval in iterations), $s$ (stride per shift).

---

## Sampling (during training — mixed ODE/SDE)

Three zones across the $T$ denoising steps:

$$\underbrace{x_T \to \cdots \to x_{l+w}}_{\text{ODE, no gradient}} \;\Big|\; \underbrace{x_{l+w} \to \cdots \to x_l}_{\text{SDE + gradient}} \;\Big|\; \underbrace{x_l \to \cdots \to x_0}_{\text{ODE, no gradient}}$$

**Outside window** (ODE):
$$x_{t-\Delta t} = x_t - v_\theta(x_t, t, c)\,\Delta t$$

**Inside window** $t \in \mathcal{W}(l)$ (SDE):
$$x_{t-\Delta t} = x_t - v_\theta(x_t,t,c)\,\Delta t + \frac{\sigma_t^2\,\Delta t}{2t^2}(\hat x_0 - x_t) + \sigma_t\sqrt{\Delta t}\,\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,I)$$

where $\hat x_0 = x_t - t\,v_\theta(x_t,t,c)$. The transition inside the window is Gaussian:
$$\pi_\theta(x_{t-\Delta t} \mid x_t, c) = \mathcal{N}\!\left(x_{t-\Delta t};\; \mu_\theta(x_t,t,c),\; \sigma_t^2\,\Delta t\, I\right), \quad t \in \mathcal{W}(l)$$

**Window sliding** (every $\tau$ training iterations):
$$l \leftarrow \min(l + s,\; T - w)$$

This rotates gradient updates through different trajectory segments over training.

---

## Reward calculation

Evaluate $r(x_0^{(i)}, c)$ at the terminal image. The ODE steps after the window run without gradient to produce $x_0$.

---

## Training objective

GRPO-clip loss restricted to window steps only:

$$\boxed{
\mathcal{L}_\text{MixGRPO}(\theta) = -\mathbb{E}\!\left[\frac{1}{N_g}\sum_{i=1}^{N_g} \frac{1}{|\mathcal{W}|}\sum_{t \in \mathcal{W}(l)} \min\!\left(\rho_t^{(i)}\hat A^{(i)},\; \text{clip}\!\left(\rho_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon\right)\hat A^{(i)}\right)\right]
}$$

Importance ratio (only window steps; ODE steps have $\rho_t = 1$ trivially):
$$\rho_t^{(i)} = \frac{\pi_\theta(x_{t-\Delta t}^{(i)} \mid x_t^{(i)}, c)}{\pi_{\theta_\text{old}}(x_{t-\Delta t}^{(i)} \mid x_t^{(i)}, c)} = \exp\!\left(-\frac{\|x_{t-\Delta t}^{(i)} - \mu_\theta\|^2 - \|x_{t-\Delta t}^{(i)} - \mu_{\theta_\text{old}}\|^2}{2\,\sigma_t^2\,\Delta t}\right), \quad t \in \mathcal{W}(l)$$

---

## MixGRPO-Flash variant

Shrink the window to **one step** ($w=1$) and use DPM-Solver++ for all ODE steps with compression rate $\tilde r < 1$:

$$\tilde T = 1 + (T - 1)\,\tilde r \quad (\text{effective total steps after compression})$$

Speedup factor relative to full MixGRPO:
$$S = T / \tilde T \approx 1 / \tilde r \quad \text{for large } T$$

Achieves **71% training time reduction** with comparable reward performance.

---

## Training algorithm

```
Input: pretrained v_θ, reward r, prompt dist p_c, N_g, w, τ, s, T
Initialize: l = 0
Repeat:
  Every τ iters:  l ← min(l+s, T-w)
  1. Sample prompts {c_j}
  2. For each c_j, generate N_g trajectories:
       x_T ~ N(0,I) per sample
       ODE from T down to l+w+1  (no grad)
       SDE from l+w down to l    (store for gradient)  ← W(l) steps
       ODE from l-1 down to 0    (no grad) → x_0^(i)
  3. R^(i) = r(x_0^(i), c_j);  Â^(i) = normalise
  4. For K gradient steps:
       ρ_t^(i) = π_θ / π_{θ_old}  for t ∈ W(l)
       L = -mean [ min(ρ·Â, clip(ρ,1-ε,1+ε)·Â) ]
       θ ← θ - η ∇_θ L
  5. θ_old ← θ
```

---

## Comparison

| Method | SDE steps / trajectory | Gradient steps | ODE solver quality |
|---|---|---|---|
| FlowGRPO | All $T$ | All $T$ | Standard |
| FlowGRPO-Fast | 1–2 (random branch point) | 1–2 | Standard |
| MixGRPO | Window $w$ (sliding) | $w$ | Standard outside window |
| MixGRPO-Flash | 1 | 1 | DPM-Solver++ outside window |

---

## Limitations

| Problem | Note |
|---|---|
| Window width $w$ is a hyperparameter | Optimal: $w=4$, $\tau=25$, $s=1$ for $T=25$ per paper ablation |
| SDE noise within window still causes artifacts | [CPS](cps.md) can be applied inside $\mathcal{W}(l)$ |
| Ratio imbalance within window still exists | [GRPO-Guard](grpo_guard.md) compatible |
