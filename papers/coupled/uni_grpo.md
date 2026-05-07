# UniGRPO — Unified Policy Optimization for Reasoning-Driven Visual Generation

> Notation: follows [NOTATION.md](../NOTATION.md). Flow matching: $v_\theta$, $t \in [0,1]$. Text policy: $\pi_\theta^\text{txt}$ over token sequence $y$. Image policy: $\pi_\theta^\text{img}$ over denoising trajectory. RatioNorm adapted from [GRPO-Guard](grpo_guard.md). Group size $G$; SDE window $T_\text{SDE} \subset \{t_1,\ldots,t_T\}$.

| Field | Value |
|---|---|
| **arXiv** | [2603.23500](https://arxiv.org/abs/2603.23500) |
| **Submitted** | 2026-03-25 |
| **Venue** | — (preprint) |
| **Authors** | Jie Liu, Zilyu Ye, Linxiao Yuan, Shenhan Zhu, Yu Gao, Jie Wu, Kunchang Li, Xionghui Wang, Xiaonan Nie, Weilin Huang, Wanli Ouyang |
| **GitHub** | — |
| **Paradigm** | **Coupled** — SDE-based sampling, per-step importance ratio for image policy; GRPO-clip for text |
| **Cites** | FlowGRPO (2505.05470), GRPO-Guard (2510.22319), CPS (2509.05952), AWM (2509.25050) |

---

## Motivation

Unified multimodal models (e.g., Transfusion, Show-o) generate both text reasoning chains and images within a single transformer. Prior RL fine-tuning approaches hit three structural problems:

1. **Fragmented optimisation**: text RL and image RL are treated as separate loops (e.g., DualGRPO); the synergy between reasoning quality and image quality is never exploited — joint MDP training is missing.
2. **CFG scalability**: classifier-free guidance requires two forward passes per step at training time, which compounds badly over long text+image rollouts.
3. **Ratio left-shift in flow models**: the importance ratio $\rho_t$ is systematically below 1 and varies ~20× in variance across timesteps (per GRPO-Guard analysis), rendering PPO clipping ineffective and causing reward hacking; standard latent-space KL penalties do not fix this.

UniGRPO addresses all three by (i) casting both text and image generation as a **single unified MDP** with shared rewards, (ii) applying RatioNorm (from GRPO-Guard) to the image policy, and (iii) replacing latent KL regularisation with **velocity-space MSE** against the reference model.

---

## Setting

- **Model**: a unified transformer $f_\theta$ that produces both text tokens and flow-matching velocity predictions. The model acts as both $\pi_\theta^\text{txt}$ (text policy) and $\pi_\theta^\text{img}$ (image policy, via SDE-converted denoising).
- **Online**: sample $G$ complete trajectories (reasoning chain $y^{(i)}$ + image $x_0^{(i)}$) per prompt $c$ each iteration.
- **Reward**: $R^{(i)} = R(x_0^{(i)}, y^{(i)}, c)$ — a terminal reward evaluated on the full (text, image) output.

---

## Sampling (training rollout)

### Text phase

The model autoregressively generates $G$ reasoning chains conditioned on prompt $c$:
$$y_k^{(i)} \sim \pi_\theta^\text{txt}(\cdot \mid c,\, y_{<k}^{(i)})$$

### Image phase (conditioned on reasoning chain)

Each reasoning chain $y^{(i)}$ conditions the flow model. The SDE is applied only over the window $T_\text{SDE}$ (following MixGRPO / FlowGRPO-Fast); ODE steps outside the window use no-gradient fast sampling:

$$\Delta x_{t_k} = \left[v_\theta(x_{t_k},t_k,c,y^{(i)}) + \frac{\sigma_{t_k}^2}{2t_k^2}\!\left(x_{t_k} + (1-t_k)v_\theta\right)\right]\Delta t + \sigma_{t_k}\sqrt{\Delta t}\,\epsilon$$

for $t_k \in T_\text{SDE}$; pure ODE elsewhere. The Gaussian transition for window steps:
$$\pi_\theta^\text{img}(x_{t_k-\Delta t} \mid x_{t_k}, c, y^{(i)}) = \mathcal{N}\!\left(x_{t_k-\Delta t};\,\mu_\theta(x_{t_k},t_k,c,y^{(i)}),\,\sigma_{t_k}^2\Delta t\,I\right)$$

---

## Reward and advantage

Group-relative advantage over the $G$ trajectories per prompt:
$$\hat A^{(i)} = \frac{R^{(i)} - \overline R}{\text{std}(\{R^{(j)}\}) + \delta}$$

---

## Training objective

### Text objective

Standard GRPO-clip applied token-by-token over the reasoning chain:

$$J_\text{Text}(\theta) = \frac{1}{G}\sum_{i=1}^G \frac{1}{|y^{(i)}|}\sum_k \left[\min\!\left(r_{i,k}^{\text{txt}}\hat A^{(i)},\;\text{clip}(r_{i,k}^{\text{txt}}, 1{-}\epsilon, 1{+}\epsilon)\hat A^{(i)}\right) - \beta_\text{txt}\,D_\text{KL}(\pi_\theta^\text{txt}\|\pi_\text{ref}^\text{txt})\right]$$

where $r_{i,k}^\text{txt} = \pi_\theta^\text{txt}(y_k^{(i)} \mid \cdot) / \pi_{\theta_\text{old}}^\text{txt}(y_k^{(i)} \mid \cdot)$ is the per-token importance ratio.

### Image objective with RatioNorm

Raw importance ratio for SDE steps (same formula as FlowGRPO):
$$\log r_{t_k}^{(i)} = -\frac{\|x_{t_k-\Delta t}^{(i)} - \mu_\theta\|^2 - \|x_{t_k-\Delta t}^{(i)} - \mu_{\theta_\text{old}}\|^2}{2\sigma_{t_k}^2\Delta t}$$

**RatioNorm** (see [GRPO-Guard](grpo_guard.md)) standardises the per-timestep distribution:
$$\log \tilde r_{t_k}^{(i)} = \sigma_{t_k}\sqrt{\Delta t}\!\left(\log r_{t_k}^{(i)} + \frac{\|\Delta\mu_{t_k}\|^2}{2\sigma_{t_k}^2\Delta t}\right) = -\Delta\mu_{t_k} \cdot \epsilon_{t_k}$$

This restores $\mathbb{E}[\log\tilde r_{t_k}] \approx 0$, $\text{std}(\log\tilde r_{t_k}) \approx 1$, making the PPO clip band $[1-\epsilon, 1+\epsilon]$ effective.

The image objective (restricted to $T_\text{SDE}$):

$$J_\text{Flow}(\theta) = \frac{1}{G}\sum_{i=1}^G \frac{1}{|T_\text{SDE}|}\sum_{t_k \in T_\text{SDE}} \left[\min\!\left(\tilde r_{t_k}^{(i)}\hat A^{(i)},\;\text{clip}(\tilde r_{t_k}^{(i)}, 1{-}\epsilon, 1{+}\epsilon)\hat A^{(i)}\right) - \beta_\text{img}\,D_\text{KL}(\pi_\theta^\text{img}\|\pi_\text{ref}^\text{img})\right]$$

### Velocity-space regularisation

Standard latent-space KL is insufficient because it does not penalise velocity-field drift directly. UniGRPO adds an MSE between the current and reference velocity fields, evaluated at forward-noised window states:

$$\mathcal{L}_\text{MSE}(\theta) = \mathbb{E}_{t_k \in T_\text{SDE},\,i}\!\left[\left\|v_\theta(x_{t_k}^{(i)},t_k,c,y^{(i)}) - v_{\theta_\text{ref}}(x_{t_k}^{(i)},t_k,c,y^{(i)})\right\|^2\right]$$

### Combined objective

$$\boxed{J_\text{UniGRPO}(\theta) = J_\text{Text}(\theta) + \lambda\,J_\text{Flow}(\theta) - \gamma\,\mathcal{L}_\text{MSE}(\theta)}$$

where $\lambda = 1$ and $\gamma$ is a small penalty weight (default 0.1).

---

## Training algorithm

```
Input: unified model θ, reward R, prompt dist p_c, group size G, SDE window T_SDE
Initialize: θ_old ← θ, θ_ref ← θ (frozen reference)
Repeat:
  1. Sample prompts {c_j}
  2. For each c_j, roll out G trajectories:
       a. Text phase: y^(i) ~ π_θ^txt(· | c_j)  [autoregressive]
       b. Image phase (conditioned on y^(i)):
            ODE steps outside T_SDE  (no gradient)
            SDE steps inside T_SDE   (store transitions)
  3. Compute terminal rewards: R^(i) = R(x_0^(i), y^(i), c_j)
  4. Group advantages: Â^(i) = (R^(i) - mean) / std

  5. Text gradient:
       r_k^txt = π_θ^txt / π_{θ_old}^txt  [per token]
       J_Text = mean[min(r^txt·Â, clip(r^txt,1-ε,1+ε)·Â) - β_txt·KL]

  6. Image gradient (SDE window):
       log ρ̃_{t_k} = -Δμ_{t_k}·ε_{t_k}     ← RatioNorm
       ρ̃_{t_k} = exp(log ρ̃_{t_k})
       J_Flow = mean[min(ρ̃·Â, clip(ρ̃,1-ε,1+ε)·Â) - β_img·KL]

  7. Velocity regularisation:
       L_MSE = mean ||v_θ(x_t^(i),t_k,·) - v_{θ_ref}(x_t^(i),t_k,·)||²

  8. Combined: J = J_Text + λ·J_Flow - γ·L_MSE
  9. θ ← θ + η ∇_θ J
  10. θ_old ← θ
```

---

## Comparison to related methods

| Aspect | FlowGRPO | GRPO-Guard | DualGRPO | UniGRPO |
|---|---|---|---|---|
| Modality | Image only | Image only | Text + Image (separate) | Text + Image (unified MDP) |
| Importance ratio | Raw $\rho_t$ | RatioNorm $\hat\rho_t$ | Separate text/image | RatioNorm for image |
| Regularisation | KL (latent) | KL (latent) | Separate KL | Velocity-space MSE |
| Reasoning chain | None | None | Separate text model | Joint, reward-guided |
| SDE scope | All $T$ steps | All $T$ steps | All $T$ steps | Window $T_\text{SDE}$ |

---

## Limitations

- Requires a unified model architecture (single transformer for text and image); not applicable to separate encoder-decoder systems.
- The velocity-space MSE regularisation computes $v_{\theta_\text{ref}}$ at each training step — doubling the image forward-pass cost relative to no regularisation.
- Reward must cover both text quality (reasoning) and image quality; misspecified rewards may trade one off against the other.
- RatioNorm is applied per group, not per population — the normalisation is approximate when group size $G$ is small.
