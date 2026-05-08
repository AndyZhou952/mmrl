# DanceGRPO — Unleashing GRPO on Visual Generation

> Notation: follows [NOTATION.md](../NOTATION.md). Uses $t$ decreasing (denoising direction). For diffusion: $\epsilon_\theta$, DDPM schedule; for flow: $v_\theta$, linear interpolation. SDE noise coefficient: $\eta_t$ (diffusion) or $\varepsilon_t$ (flow).

| Field | Value |
|---|---|
| **arXiv** | [2505.07818](https://arxiv.org/abs/2505.07818) |
| **Submitted** | 2025-05-12 (revised 2025-08-28) |
| **Venue** | — (preprint) |
| **Authors** | Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, Ping Luo |
| **GitHub** | https://github.com/XueZeyue/DanceGRPO |
| **Project** | https://dancegrpo.github.io |
| **Paradigm** | **Coupled** — per-step Gaussian log-prob required; SDE-based sampling |
| **Cites** | DDPO, DPOK, GRPO (DeepSeek-R1), SD3, FLUX, HunyuanVideo, SkyReels-I2V |
| **Cited by** | CPS, DiffusionNFT, GRPO-Guard, DGPO |

---

## Motivation

Concurrent with FlowGRPO (4 days later), DanceGRPO independently arrives at the SDE-conversion insight but frames it as a **unified framework**: a single GRPO algorithm should work for both diffusion models (DDPM-style) and rectified flows, across T2I, T2V, and I2V tasks. The paper shows that existing methods (DDPO, DPOK) are unstable at scale and cannot handle diverse prompt sets, while DanceGRPO generalises to video and handles sparse binary rewards.

---

## Setting

- **Two paradigms**: DDPM-style diffusion AND rectified flow.
- **Three tasks**: text-to-image (T2I), text-to-video (T2V), image-to-video (I2V).
- **Four backbones**: Stable Diffusion, FLUX, HunyuanVideo, SkyReels-I2V.
- **Five reward types**: image aesthetics, video aesthetics, text-image alignment, video motion quality, binary reward.

---

## Sampling (inference)

For each modality, standard deterministic sampling:
- **T2I / T2V** (flow models): ODE from $t=1$ to $t=0$.
- **I2V**: condition on reference frame, ODE from $t=1$ to $t=0$.

---

## Key contribution — Unified SDE for both paradigms

### A. DDPM-style diffusion — Reverse SDE

The standard DDPM reverse process $p_\theta(x_{t-1} \mid x_t)$ is already stochastic. DanceGRPO writes it as a continuous SDE:

$$d\mathbf{z}_t = \left(f_t \mathbf{z}_t - \frac{1+\eta_t^2}{2}\,g_t^2\,\nabla \log p_t(\mathbf{z}_t)\right) dt + \eta_t\,g_t\, d\mathbf{W}_t$$

where $f_t$ and $g_t$ are the drift/diffusion coefficients of the forward process, $\eta_t \in [0,1]$ controls stochasticity ($\eta_t=0$: deterministic DDIM; $\eta_t=1$: full DDPM).

Score approximation via noise prediction:
$$\nabla \log p_t(\mathbf{z}_t) \approx -\epsilon_\theta(\mathbf{z}_t, t, c) / \sigma_t$$

Euler-Maruyama discretisation gives a Gaussian step:
$$\pi_\theta(\mathbf{z}_{t-1} \mid \mathbf{z}_t, c) = \mathcal{N}\left(\mathbf{z}_{t-1};\, \mu_\theta^\text{DDPM}(\mathbf{z}_t, t, c),\, \eta_t^2 g_t^2 \Delta t\, I\right)$$

### B. Rectified flow — Reverse SDE

The flow ODE $d\mathbf{z}_t = v_\theta\, dt$ is converted to an SDE by adding controlled noise:

$$d\mathbf{z}_t = \left(v_\theta(\mathbf{z}_t,t,c) - \frac{\varepsilon_t^2}{2}\,\nabla \log p_t(\mathbf{z}_t)\right) dt + \varepsilon_t\, d\mathbf{W}_t$$

For the linear interpolation path $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\epsilon$:
$$\nabla \log p_t(\mathbf{z}_t) = \frac{\hat{\mathbf{z}}_0(\mathbf{z}_t,t) - \mathbf{z}_t}{t^2}, \quad \hat{\mathbf{z}}_0 = \mathbf{z}_t - t\,v_\theta(\mathbf{z}_t,t,c)$$

Discretisation:
$$\pi_\theta(\mathbf{z}_{t-\Delta t} \mid \mathbf{z}_t, c) = \mathcal{N}\left(\mathbf{z}_{t-\Delta t};\, \mu_\theta^\text{flow}(\mathbf{z}_t,t,c),\, \varepsilon_t^2\,\Delta t\, I\right)$$

Both cases yield tractable **Gaussian per-step likelihoods** with the same functional form, enabling a single unified importance ratio:
$$\rho_{t}^{(i)} = \frac{\pi_\theta(\mathbf{z}_{t-1}^{(i)} \mid \mathbf{z}_t^{(i)}, c)}{\pi_{\theta_\text{old}}(\mathbf{z}_{t-1}^{(i)} \mid \mathbf{z}_t^{(i)}, c)} = \exp\left(-\frac{\Vert\mathbf{z}_{t-1}^{(i)} - \mu_\theta\Vert^2 - \Vert\mathbf{z}_{t-1}^{(i)} - \mu_{\theta_\text{old}}\Vert^2}{2\,\sigma_\text{SDE}^2}\right)$$

---

## Reward calculation

DanceGRPO uses up to 5 reward models simultaneously (multi-reward):
- Image aesthetics: HPS-v2.1 or similar
- Text-image alignment: CLIP score
- Video motion quality: VideoAlign
- Video aesthetics: separate video aesthetic predictor
- Binary reward (0/1): e.g., safety filter or hard constraint

Multi-reward: scalar rewards are combined (weighted sum or Pareto weighting).

---

## Training objective

Group advantage (same as FlowGRPO):
$$\hat A^{(i)} = \frac{r^{(i)} - \overline r}{\text{std}(\{r^{(j)}\}) + \delta}$$

PPO-clipped GRPO objective; DanceGRPO **subsamples** $\lceil \tau T \rceil$ timesteps (default $\tau = 0.6$) to reduce memory:

$$\boxed{
\mathcal{L}_\text{DanceGRPO}(\theta) = -\mathbb{E}\left[\frac{1}{N_g}\sum_{i=1}^{N_g} \frac{1}{\lceil\tau T\rceil}\sum_{t \in \mathcal{T}_\text{sub}} \min\left(\rho_t^{(i)}\hat A^{(i)},\; \text{clip}\left(\rho_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon\right)\hat A^{(i)}\right)\right]
}$$

where $\mathcal{T}_\text{sub}$ is the randomly-sampled subset of $\lceil\tau T\rceil$ timesteps.

---

## Training algorithm

```
Input: pretrained model θ, reward models {r_k}, prompt dist p_c, group size N_g, τ
Repeat:
  1. Sample prompts {c_j}
  2. For each c_j, generate N_g trajectories via SDE (modality-appropriate):
       {z_T^(i),...,z_0^(i)} for i=1,...,N_g
       Store sampled actions z_{t-1}^(i) for all t
  3. Compute scalar rewards: R^(i) = sum_k w_k r_k(z_0^(i), c_j)
  4. Compute group advantages: Â^(i) = (R^(i) - mean) / std
  5. Subsample ⌈τT⌉ timesteps: T_sub ⊂ {1,...,T}
  6. For K gradient steps:
       For t ∈ T_sub, i=1,...,N_g:
         ρ_t^(i) = π_θ(z_{t-1}^(i)|z_t^(i),c) / π_{θ_old}(...)
       L = -mean[min(ρ·Â, clip(ρ,1-ε,1+ε)·Â)]
       θ ← θ - η ∇_θ L
  7. θ_old ← θ
```

---

## Key difference from FlowGRPO

| Aspect | FlowGRPO | DanceGRPO |
|---|---|---|
| Modality | T2I only | T2I + T2V + I2V |
| Paradigm support | Flow matching only | DDPM + rectified flow |
| Timestep subsampling | Denoising reduction (fixed $T_\text{train}$) | Random $\tau$-fraction of all steps |
| Efficiency variant | Flow-GRPO-Fast | — |
| Binary reward | Not demonstrated | Demonstrated stable |
| Video | — | HunyuanVideo, SkyReels |

The two SDE derivations differ slightly in notation but are equivalent in spirit. CPS identifies that both suffer from the same noise-artifact problem.

---

## Limitations

| Problem | Addressed by |
|---|---|
| SDE noise → artifacts | [CPS](cps.md) |
| Ratio imbalance → reward hacking | [GRPO-Guard](grpo_guard.md) |
| No efficiency variant | [MixGRPO](mix_grpo.md) |
| SDE blocks ODE samplers | [DGPO](../decoupled/dgpo.md) |
