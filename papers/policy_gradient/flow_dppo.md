# FlowDPPO — Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models

| Field | Value |
|---|---|
| **arXiv** | [2606.11025](https://arxiv.org/abs/2606.11025) |
| **Submitted** | 2026-06-09 |
| **Venue** | — (preprint) |
| **Authors** | Bowen Ping, Xiangxin Zhou, Penghui Qi, Minnan Luo, Liefeng Bo, Tianyu Pang |
| **GitHub** | https://github.com/Tencent-Hunyuan/UniRL (`unirl/algorithms/flowdppo.py`) |
| **Paradigm** | **Policy Gradient** — extends FlowGRPO; PPO ratio clipping replaced by an exact per-step Gaussian-KL divergence mask over SDE transitions |
| **Cites** | FlowGRPO (2505.05470), CPS (2509.05952), GRPO-Guard (2510.22319), PPO, GRPO |
| **Cited by** | — |

---

## Context

Flow-DPPO is a direct **add-on to [FlowGRPO](flow_grpo.md)**: it keeps FlowGRPO's stochastic reverse-SDE rollout, group-relative advantage $\hat{A}^{(i)}$, and per-step importance ratio $\rho_t^{(i)}$ unchanged. The **only** thing it changes is *how the trust region is enforced* — replacing PPO-style ratio clipping with a divergence proximal constraint. It positions itself against FlowGRPO and CPS (which both rely on ratio clipping) and is motivated by the same ratio-pathology that [GRPO-Guard](grpo_guard.md) diagnoses, but resolves it differently: GRPO-Guard re-centres the ratio so the clip works; Flow-DPPO discards the clip entirely.

---

## Problem 1 — Ratio clipping is a noisy single-sample proxy for the trust region

**Issue**: FlowGRPO and CPS cast denoising as an MDP and use PPO ratio clipping $\mathrm{clip}(\rho_t, 1{-}\epsilon, 1{+}\epsilon)$ to approximate a trust region. But at each denoising step $\rho_t^{(i)}$ is a **single-sample** estimate of the policy ratio drawn from a high-dimensional Gaussian transition. As an estimator of the true policy divergence it is high-variance: the same nominal clip band over-constrains updates in some regions of the trajectory and under-constrains them in others (the left-shift / cross-timestep variance documented by [GRPO-Guard](grpo_guard.md)).

**Idea**: Replace the ratio-clip proxy with a **divergence proximal constraint** computed from the *exact* per-step KL divergence between the rollout policy and the current policy, and gate updates with an **asymmetric mask** rather than clipping the ratio.

**Why this works**: The reverse-SDE per-step policy is Gaussian, $\pi_\theta(x_{t-\Delta t}\mid x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$, and the rollout ($\theta_\text{old}$) and replayed ($\theta$) policies share the **same** variance $\sigma_t^2$ (it is fixed by the SDE schedule, not learned). The KL between two equal-variance Gaussians is therefore available in closed form from the means alone — no sampling, no noise:

$$D_t^{(i)} = \mathrm{KL}\left(\pi_{\theta_\text{old}} \Vert \pi_\theta\right) = \frac{\Vert\mu_\theta(x_t^{(i)}) - \mu_{\theta_\text{old}}(x_t^{(i)})\Vert^2}{2\sigma_t^2}$$

This is an exact, cheap per-step trust-region measure. The mask is **asymmetric**: it blocks an update only when it *both* violates the divergence threshold *and* is still moving **away** from the rollout policy in the advantage direction — exactly the updates clipping is meant to stop. Updates that pull **back** toward the rollout policy (corrective) or stay within the threshold keep their gradient:

$$\text{mask out } (i,t) \iff D_t^{(i)} > \epsilon_D \text{ and } \begin{cases} \rho_t^{(i)} > 1 & \text{if } \hat{A}^{(i)} > 0 \\ \rho_t^{(i)} < 1 & \text{if } \hat{A}^{(i)} < 0 \end{cases}$$

where the two inherited FlowGRPO quantities are the **per-step importance ratio** $\rho_t^{(i)} = \dfrac{\pi_\theta(x_{t-\Delta t}^{(i)}\mid x_t^{(i)}, c)}{\pi_{\theta_\text{old}}(x_{t-\Delta t}^{(i)}\mid x_t^{(i)}, c)}$ — the current policy $\pi_\theta$ over the frozen rollout policy $\pi_{\theta_\text{old}}$ that generated the trajectory — and the **group-relative advantage** $\hat{A}^{(i)} = \dfrac{r^{(i)} - \mathrm{mean}(\lbrace r^{(j)}\rbrace)}{\mathrm{std}(\lbrace r^{(j)}\rbrace) + \delta}$, sample $i$'s reward $r^{(i)}$ standardised within its group of $N$ images from one prompt $c$ ($\delta > 0$ guards the denominator; it is shared across all steps and detached). $\epsilon_D > 0$ is the scalar **divergence threshold** (a hyperparameter, default $\approx 10^{-5}$). So a step's gradient is dropped only when the update has drifted past $\epsilon_D$ *and* is still pushing **away** from the rollout policy in the advantage direction.

**Result**: The paper reports that Flow-DPPO "achieves higher rewards with better KL-proximal efficiency, alleviates catastrophic forgetting, promotes balanced multi-objective optimization, and enables stable multi-epoch training where ratio clipping degrades" (abstract). In VeRL-Omni it is validated by LoRA post-training of Qwen-Image on the OCR task (zero-std-ratio and reward curves in the VeRL-Omni performance reference).

---

## Training Objective

Same group-relative policy gradient as FlowGRPO, but each per-step term is gated by the divergence keep-mask $m_t^{(i)}$ instead of the ratio being clipped (and there is no separate KL penalty term — the trust region *is* the mask):

$$\boxed{
\mathcal{L}_\text{FlowDPPO}(\theta) = -\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\frac{1}{T}\sum_{t=1}^{T} m_t^{(i)}\rho_t^{(i)}\hat{A}^{(i)}\right]
}$$

Reading the box: the sum runs over the $N$ samples in a group and the $T$ denoising steps of each rollout; $\rho_t^{(i)}$ (per-step importance ratio) and $\hat{A}^{(i)}$ (group-relative advantage) are exactly as defined in Problem 1. The one new factor is the **divergence keep-mask** $m_t^{(i)} \in \lbrace 0, 1\rbrace$ — the trust region itself, multiplying each per-step term by $0$ (drop) or $1$ (keep) in place of PPO's $\mathrm{clip}(\rho_t, 1{-}\epsilon, 1{+}\epsilon)$. It is the indicator of the Problem 1 mask condition:

$$m_t^{(i)} = \begin{cases} 0 & D_t^{(i)} > \epsilon_D \text{ and } \big[(\hat{A}^{(i)}>0 \wedge \rho_t^{(i)}>1) \vee (\hat{A}^{(i)}<0 \wedge \rho_t^{(i)}<1)\big] \\ 1 & \text{otherwise} \end{cases}$$

The mask is detached — a hard gate, not a differentiable penalty — so blocked steps contribute exactly zero gradient while corrective steps (those pulling back toward $\pi_{\theta_\text{old}}$) and within-threshold steps keep their full gradient.

---

## Reference Implementation (VeRL-Omni)

Condensed from [`FlowDPPOLoss` in `diffusion_algos.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/diffusion/diffusion_algos.py) (`@register_diffusion_loss("flow_dppo")`). Shown as a **diff against `FlowGRPOLoss`** — the `# <<<` lines are the entire change: there is no `clip`/`max`; instead the per-element loss is the unclipped `-adv*ratio` zeroed by the asymmetric KL mask. Extra inputs `prev_sample_mean`, `old_prev_sample_mean`, `std_dev_t`, `sqrt_dt` come from the rollout (`add_kl_coefficient=True` divides by the SDE noise scale; `kl_mask_threshold` is $\epsilon_D$, e.g. `1e-5`):

```python
@register_diffusion_loss("flow_dppo")
def loss_flow_dppo(old_lp, lp, adv, mean_θ, mean_old, std_t, sqrt_dt, cfg):
    c = cfg.diffusion_loss
    adv = adv.detach()
    ratio = exp(lp - old_lp)
    unclipped = -adv * ratio                                       # FlowGRPO: max(unclipped, clipped)
    sigma_t = std_t * sqrt_dt                                      # <<< SDE noise scale (add_kl_coefficient)
    kl = ((mean_θ - mean_old) ** 2).mean(non_batch_dims) / (2 * sigma_t ** 2)  # <<< exact Gaussian KL
    high_kl = kl >= c.kl_mask_threshold                            # <<< divergence trust region (ε_D)
    block = high_kl & (((ratio > 1) & (adv > 0)) | ((ratio < 1) & (adv < 0)))  # <<< asymmetric mask
    per_elem = where(~block, unclipped, 0.0)                       # <<< mask replaces PPO clip
    return mean(per_elem)
```

---

## Limitations

| Problem | Note |
|---|---|
| Divergence threshold $\epsilon_D$ is a hyperparameter | Default `kl_mask_threshold=1e-5`; trades off exploration vs. proximity |
| Mask is a hard binary gate | No soft/annealed penalty; a transition is either fully kept or fully dropped |
| Requires caching rollout transition means $\mu_{\theta_\text{old}}$ | Extra per-step rollout state vs. FlowGRPO (which needs only old log-probs) |
| Still requires the SDE rollout | Same cost as FlowGRPO; efficiency fixes ([MixGRPO](mix_grpo.md)) and sample-quality fixes ([CPS](cps.md)) apply orthogonally |
