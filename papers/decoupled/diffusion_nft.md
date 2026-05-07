# DiffusionNFT — Negative Flow Training for Diffusion RL

> Notation: follows [NOTATION.md](../NOTATION.md) §3 (flow matching convention). Flow model: $v_\theta$; forward path $x_t = (1-t)x_0 + t\epsilon$; velocity target $u_t = x_0 - \epsilon$. Old policy: $v_{\theta_\text{old}}$, updated via EMA. Reward $r \in [0,1]$ (normalised).

| Field | Value |
|---|---|
| **arXiv** | [2509.16117](https://arxiv.org/abs/2509.16117) |
| **Submitted** | 2025-09-25 (revised 2026-02) |
| **Venue** | — (preprint) |
| **Authors** | (see paper) |
| **Paradigm** | **Decoupled** — forward-process loss; no SDE, no importance ratio, any ODE sampler |
| **Cites** | FlowGRPO (2505.05470), DanceGRPO (2505.07818), DDPO, flow matching |

---

## Motivation

Coupled methods (FlowGRPO, DanceGRPO) treat RL as a reverse-process MDP: they optimise the policy at each *denoising* step. Three structural problems follow:

1. **Solver lock-in**: importance ratio $\rho_t = \pi_\theta / \pi_{\theta_\text{old}}$ forces SDE sampling; fast ODE solvers are excluded.
2. **Forward–reverse inconsistency**: the reward is computed on $x_0$, but gradients flow through the reverse chain; the $x_0$ distribution during RL differs from pretraining.
3. **CFG conflict**: classifier-free guidance perturbs the reverse process in ways that are incompatible with the per-step likelihood in GRPO.

DiffusionNFT sidesteps all three by moving to the **forward (noising) direction**. The training signal is injected into the standard flow matching objective — which already sums over all noise levels $t$ — without touching the reverse chain.

---

## Setting

- **Model**: flow matching model $v_\theta$ with old policy reference $v_{\theta_\text{old}}$ (maintained via EMA).
- **Online**: generate $N$ images per prompt each iteration with **any ODE sampler**.
- **Reward**: $r^{(i)} \in [0,1]$ (normalised per batch; high = positive, low = negative).
- **Coupling parameter** $\beta \in (0,1]$: controls how far the implicit policies deviate from $v_{\theta_\text{old}}$.

---

## Sampling (inference)

Any deterministic ODE sampler from $t=1$ to $t=0$:
$$x_{t-\Delta t} = x_t - v_\theta(x_t, t, c)\,\Delta t$$

No SDE required. The sampler choice does not affect the training objective.

---

## Reward calculation

Evaluate reward on the generated images $\{x_0^{(i)}\}$:
$$r^{(i)} = r(x_0^{(i)}, c), \quad r^{(i)} \in [0,1]$$

Normalise reward into per-batch weights (normalised to $[0,1]$ via min–max or rank-based rescaling so that $r=1$ is fully positive and $r=0$ is fully negative).

The complement $(1 - r^{(i)})$ acts as the negative weight for the same image.

---

## Training objective

### Step 1 — Implicit positive and negative policies

Rather than explicitly sampling from a reward-weighted distribution, DiffusionNFT defines **implicit positive/negative velocity fields** as linear interpolations around $v_{\theta_\text{old}}$:

$$v_\theta^+(x_t, t, c) = (1-\beta)\,v_{\theta_\text{old}}(x_t,t,c) + \beta\, v_\theta(x_t,t,c)$$
$$v_\theta^-(x_t, t, c) = (1+\beta)\,v_{\theta_\text{old}}(x_t,t,c) - \beta\, v_\theta(x_t,t,c)$$

**Intuition**:
- $v_\theta^+$ moves in the direction of the current update ($\theta - \theta_\text{old}$) — it represents the "improved" policy.
- $v_\theta^-$ moves in the *opposite* direction — it represents the "degraded" policy.

These are proxies for $\pi^+(x_0|c) \propto r \cdot \pi_{\theta_\text{old}}(x_0|c)$ and $\pi^-(x_0|c) \propto (1-r) \cdot \pi_{\theta_\text{old}}(x_0|c)$, respectively.

### Step 2 — Forward-noised training targets

For each generated image $x_0^{(i)}$, construct forward-noised versions at random $t$:
$$x_t^{(i)} = (1-t)\,x_0^{(i)} + t\,\epsilon^{(i)}, \quad \epsilon^{(i)} \sim \mathcal{N}(0,I)$$

The clean velocity target is:
$$u_t^{(i)} = x_0^{(i)} - \epsilon^{(i)}$$

### Step 3 — Contrastive flow matching loss

$$\boxed{
\mathcal{L}_\text{NFT}(\theta) = \mathbb{E}_{t,\epsilon}\!\left[\frac{1}{N}\sum_{i=1}^N \left(r^{(i)}\,\left\|v_\theta^+(x_t^{(i)},t,c) - u_t^{(i)}\right\|^2 + (1{-}r^{(i)})\,\left\|v_\theta^-(x_t^{(i)},t,c) - u_t^{(i)}\right\|^2\right)\right]
}$$

Substituting the implicit policy definitions:
$$= \mathbb{E}\!\left[\sum_i \left(r^{(i)}\,\|(1{-}\beta)v_\text{old} + \beta v_\theta - u_t^{(i)}\|^2 + (1{-}r^{(i)})\,\|(1{+}\beta)v_\text{old} - \beta v_\theta - u_t^{(i)}\|^2\right)\right]$$

**Why this works**: minimising over $v_\theta$ pushes $v_\theta^+$ toward the target for *high*-reward images and $v_\theta^-$ toward the target for *low*-reward images. Because $v_\theta^+$ is a positive perturbation of $v_\text{old}$ and $v_\theta^-$ is a negative perturbation, training jointly causes $v_\theta$ to deviate from $v_\text{old}$ in the direction that increases reward.

**Connection to pretraining**: when all $r^{(i)} = 0.5$ (uniform reward), the loss reduces to a symmetric regression toward $u_t$ with both implicit policies averaging back to $v_\text{old}$ — equivalent to the pretraining flow matching loss. RL signal enters only through asymmetric $r^{(i)}$ values.

---

## Reference policy update

Because $v_{\theta_\text{old}}$ appears inside the loss, it must track $v_\theta$ over training. A scheduled EMA update:
$$\theta_\text{old} \leftarrow \eta_i\,\theta_\text{old} + (1-\eta_i)\,\theta$$

where $\eta_i \to 1$ as training progresses (warm-up to near-identity update). This keeps $v_\theta^{\pm}$ meaningful perturbations rather than arbitrary functions.

---

## Training algorithm

```
Input: pretrained v_θ, reward r, prompt dist p_c, group size N, β, EMA schedule {η_i}
Initialize: θ_old ← θ
Repeat (iteration i):
  1. Sample prompts {c_j}
  2. For each c_j, generate N images via ODE (any fast sampler):
       x_0^(1),...,x_0^(N) ~ ODE_θ(c_j)
  3. Compute rewards: R^(k) = r(x_0^(k), c_j) ∈ [0,1]
     (normalise within batch via min-max or rank)
  4. For each training batch (t, ε):
       t ~ Uniform[0,1];   ε^(k) ~ N(0,I)
       x_t^(k) = (1-t)·x_0^(k) + t·ε^(k)   ← forward noising
       u_t^(k) = x_0^(k) - ε^(k)            ← clean target
       v^+ = (1-β)·v_{θ_old}(x_t^(k),t,c) + β·v_θ(x_t^(k),t,c)
       v^- = (1+β)·v_{θ_old}(x_t^(k),t,c) - β·v_θ(x_t^(k),t,c)
       L = mean_k [ R^(k)||v^+ - u_t^(k)||² + (1-R^(k))||v^- - u_t^(k)||² ]
  5. θ ← θ - η ∇_θ L
  6. EMA update: θ_old ← η_i·θ_old + (1-η_i)·θ
```

---

## Comparison to reverse-process methods

| Aspect | FlowGRPO / DanceGRPO | DiffusionNFT |
|---|---|---|
| Process direction | Reverse (denoising) | Forward (noising) |
| SDE required | Yes | No |
| Importance ratio | Per-step $\rho_t^{(i)}$ | None |
| CFG at training | Conflicts with IS ratio | Compatible (no IS ratio) |
| Objective relation to pretraining | Diverges | Near-identical base loss |
| Speedup vs. FlowGRPO | — | ~25× (reported) |
| Theoretical convergence | Policy gradient guarantees | Less established |

---

## Limitations

- Reward must be normalised to $[0,1]$ (or equivalent) to give interpretable positive/negative splits; sparse or binary rewards require careful handling.
- The implicit policy coupling parameter $\beta$ requires tuning; too large destabilises $v_\theta^{\pm}$.
- No explicit KL regularisation against the reference; relies on EMA coupling and small $\beta$ to prevent mode collapse.
- Theoretical convergence guarantees weaker than policy gradient methods.
