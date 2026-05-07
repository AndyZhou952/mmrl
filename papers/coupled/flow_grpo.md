# FlowGRPO — Flow-GRPO: Training Flow Matching Models via Online RL

> Notation: follows [NOTATION.md](../NOTATION.md). Flow matching convention: $v_\theta$, continuous $t \in [0,1]$, $x_1 \sim \mathcal{N}(0,I)$ (noise), $x_0$ (clean). SDE diffusion coefficient: $s_t$ (small hyperparameter).

| Field | Value |
|---|---|
| **arXiv** | [2505.05470](https://arxiv.org/abs/2505.05470) |
| **Submitted** | 2025-05-08 (v5: 2025-10-27) |
| **Venue** | NeurIPS 2025 |
| **Authors** | Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang |
| **GitHub** | https://github.com/yifan123/flow_grpo |
| **Paradigm** | **Coupled** — per-step Gaussian log-prob required; must use SDE sampler |
| **Cites** | DDPO, GRPO (DeepSeek-R1), PPO, flow matching (Lipman et al.), SD3.5-M, FLUX |
| **Cited by** | MixGRPO, CPS, DiffusionNFT, AWM, GRPO-Guard, DGPO |

---

## Motivation

GRPO from LLMs achieves large alignment gains via group-relative policy gradients. Applying it to flow matching models is blocked by one obstacle: flow matching uses a **deterministic ODE**, which has no density — so the importance ratio $\rho_t = \pi_\theta / \pi_{\theta_\text{old}}$ is undefined. FlowGRPO solves this by converting the ODE to an equivalent SDE that preserves the marginal distribution at every $t$, enabling tractable per-step Gaussian likelihoods.

---

## Setting

- **Model**: flow matching (rectified flow) with velocity predictor $v_\theta(x_t, t, c)$.
- **ODE** (inference): $dx_t = v_\theta(x_t, t, c)\, dt$, integrated from $t=1$ to $t=0$ with $N$ Euler steps.
- **Group size** $N_g$ (typically 8–16): number of images generated per prompt.
- **Reward model**: $r(x_0, c)$, black-box scalar.

---

## Sampling (inference after training)

Standard flow ODE (unchanged from pretrained model):
$$x_{t - \Delta t} = x_t - v_\theta(x_t, t, c)\,\Delta t, \quad t = 1, 1{-}\Delta t, \ldots, \Delta t$$
with $x_1 \sim \mathcal{N}(0,I)$.

---

## Key contribution 1 — ODE-to-SDE conversion

To enable GRPO during **training**, replace the ODE with a stochastic variant that has the **same marginal** $p_t(x_t)$ at every $t$.

Using the Fokker-Planck / continuity equation duality, the equivalent SDE is:
$$dx_t = \underbrace{\left[v_\theta(x_t,t,c) + \frac{s_t^2}{2}\,\nabla_{x_t}\log p_t(x_t)\right]}_{\text{drift}}\, dt + s_t\, dW_t$$

The score $\nabla_{x_t} \log p_t(x_t)$ is approximated via Tweedie's formula:
$$\nabla_{x_t} \log p_t(x_t) \approx \frac{\hat x_0(x_t, t) - x_t}{t^2}, \quad \hat x_0 = x_t - t\, v_\theta(x_t, t, c)$$

**Euler-Maruyama discretisation** (one step from $t$ to $t - \Delta t$):
$$x_{t-\Delta t} = x_t - v_\theta(x_t,t,c)\,\Delta t + \frac{s_t^2 \Delta t}{2t^2}\left(\hat x_0 - x_t\right) + s_t\sqrt{\Delta t}\,\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,I)$$

This step is a **Gaussian** with mean $\mu_\theta(x_t, t)$ and variance $s_t^2 \Delta t\, I$:
$$\pi_\theta(x_{t-\Delta t} \mid x_t, c) = \mathcal{N}\!\left(x_{t-\Delta t};\; \mu_\theta(x_t,t,c),\; s_t^2\,\Delta t\, I\right)$$

The log-likelihood (used in the importance ratio) is:
$$\log \pi_\theta(x_{t-\Delta t} \mid x_t, c) = -\frac{\|x_{t-\Delta t} - \mu_\theta(x_t,t,c)\|^2}{2\, s_t^2\,\Delta t} + \text{const}$$

---

## Key contribution 2 — Group advantage estimation

Generate $N_g$ images $\{x_0^{(i)}\}_{i=1}^{N_g}$ for the **same** prompt $c$ using the SDE sampler:

$$\hat A^{(i)} = \frac{r^{(i)} - \overline r}{\text{std}(\{r^{(j)}\}) + \delta}, \quad \overline r = \frac{1}{N_g}\sum_{j=1}^{N_g} r^{(j)}$$

This is the standard GRPO advantage: group-normalised, zero-mean, unit-variance (approximately). It replaces the raw reward $r^{(i)}$ used in DDPO.

---

## Training objective

PPO-clipped GRPO objective summed over all $T$ training steps:

$$\boxed{
\mathcal{L}_\text{FlowGRPO}(\theta) = -\mathbb{E}_{c,\{x_0^{(i)}\}}\!\left[\frac{1}{N_g}\sum_{i=1}^{N_g} \frac{1}{T}\sum_{t=1}^{T} \min\!\left(\rho_t^{(i)}\hat A^{(i)},\; \text{clip}\!\left(\rho_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon\right)\hat A^{(i)}\right)\right] + \beta\, D_\text{KL}(\pi_\theta \| \pi_\text{ref})
}$$

where:
$$\rho_t^{(i)} = \frac{\pi_\theta(x_{t-\Delta t}^{(i)} \mid x_t^{(i)}, c)}{\pi_{\theta_\text{old}}(x_{t-\Delta t}^{(i)} \mid x_t^{(i)}, c)} = \exp\!\left(-\frac{\|x_{t-\Delta t}^{(i)} - \mu_\theta\|^2 - \|x_{t-\Delta t}^{(i)} - \mu_{\theta_\text{old}}\|^2}{2\,s_t^2\,\Delta t}\right)$$

---

## Key contribution 3 — Denoising reduction

**Problem**: computing gradients through all $T$ SDE steps is $O(T)$ memory. Typical $T=10$ during training vs. $T=40$ at inference.

**Solution**: use a smaller $T_\text{train} \ll T_\text{inf}$. The policy is evaluated on $T_\text{train}=10$ SDE steps, giving almost the same learning signal at $1/4$ the memory cost.

### Flow-GRPO-Fast variant

Even cheaper: generate a **full ODE trajectory** first (no gradient tracking), then at one randomly chosen timestep $t^*$ branch into $N_g$ SDE samples:

$$x_{t^*} \xrightarrow{\text{ODE, no grad}} \text{(shared prefix)} \quad \xrightarrow{1\text{-step SDE}} \{x_{t^*-\Delta t}^{(i)}\}_{i=1}^{N_g}$$

Only 1–2 gradient-tracked steps per trajectory. Reward is still computed at $x_0$ (run ODE from $x_{t^*-\Delta t}^{(i)}$ to $x_0^{(i)}$ without gradient tracking). Matches reward performance at significantly reduced cost.

---

## Training algorithm

```
Input: pretrained v_θ, reward model r, prompt dist p_c, group size N_g
Repeat:
  1. Sample prompts {c_j}
  2. For each c_j, generate N_g trajectories via SDE (T_train steps):
       x_1 ~ N(0,I) [shared noise for all group members]
       For t = 1, 1-Δt, ..., Δt:
         For i = 1,...,N_g:
           μ_θ_old ← drift(v_{θ_old}, x_t^(i), t)
           x_{t-Δt}^(i) = μ_θ_old + s_t√(Δt) · ε_t^(i)
       x_0^(i) obtained at t=0
  3. Compute rewards R^(i) = r(x_0^(i), c_j)
  4. Compute group advantages Â^(i) = (R^(i) - mean) / std
  5. For K gradient steps (reusing trajectories):
       Compute ρ_t^(i) = π_θ(x_{t-Δt}^(i)|x_t^(i),c) / π_{θ_old}(...)
       L = -mean[clip-ratio × Â] + β·KL
       θ ← θ - η ∇_θ L
  6. θ_old ← θ
```

---

## Comparison to DDPO

| Aspect | DDPO | FlowGRPO |
|---|---|---|
| Model family | DDPM ($\epsilon_\theta$) | Flow matching ($v_\theta$) |
| Policy density | Native (DDPM stochastic) | ODE-to-SDE conversion |
| Advantage | Raw reward $r$ | Group-normalised $\hat A^{(i)}$ |
| Objective | IS-weighted REINFORCE / PPO | PPO-clip GRPO |
| Efficiency | All $T$ steps with grad | Denoising reduction $T_\text{train} \ll T_\text{inf}$ |

---

## Limitations

| Problem | Addressed by |
|---|---|
| SDE noise → image artifacts → misleads reward model | [CPS](cps.md) |
| Importance ratio $\rho_t$ mean $<1$, varying variance → reward hacking | [GRPO-Guard](grpo_guard.md) |
| SDE sampler blocks fast ODE; still expensive | [MixGRPO](mix_grpo.md), [AWM](../decoupled/awm.md) |
| Pretraining objective ≠ GRPO objective | [AWM](../decoupled/awm.md) |
| Requires SDE → no ODE samplers | [DGPO](../decoupled/dgpo.md) |
