# DiffusionNFT — Online Diffusion Reinforcement with Forward Process

| Field | Value |
|---|---|
| **arXiv** | [2509.16117](https://arxiv.org/abs/2509.16117) |
| **Submitted** | 2025-09-25 (revised 2026-02) |
| **Venue** | ICLR 2026 (Oral) |
| **Authors** | Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, Ming-Yu Liu |
| **GitHub** | [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) |
| **Paradigm** | **Direct Preference** — forward-process matching loss; no SDE, no importance ratio, any ODE sampler |
| **Cites** | FlowGRPO (2505.05470), DanceGRPO (2505.07818), DDPO, flow matching |
| **Cited by** | — |

---

## Context

DiffusionNFT (NFT = **Negative-aware FineTuning**) takes the most radical departure from the policy-gradient paradigm: instead of modifying the *reverse* (denoising) process to enable policy gradients, it moves the training signal entirely into the *forward* (noising) direction. The reward still comes from generated images (via an ODE sampler), but the gradient flows through the pretraining flow matching loss — not through denoising steps. Throughout, the model is a velocity-prediction network $v_\theta(x_t, t, c)$ ($\theta$ = trainable parameters, $x_t$ = noisy latent at flow time $t \in [0,1]$ with $t=1$ pure noise and $t=0$ clean, $c$ = text-prompt conditioning); this same network is what an ODE sampler integrates from noise to image. This makes the approach compatible with any ODE sampler and CFG.

---

## Problem — Reverse-process RL forces SDE sampling, conflicts with CFG, and diverges from pretraining

**Issue**: Policy Gradient methods (FlowGRPO, DanceGRPO) treat RL as a reverse-process MDP: the policy is the denoising chain $x_T \to \cdots \to x_0$, and gradients flow through reverse steps via importance ratios. Three structural problems arise:

1. **Solver lock-in**: The importance ratio $\rho_t = \pi_\theta / \pi_{\theta_\text{old}}$ requires a *stochastic* policy — it forces SDE sampling and blocks the fast ODE solvers used in production.
2. **CFG conflict**: Classifier-free guidance applies to the reverse process in a way that is incompatible with the per-step likelihood used in GRPO (CFG modifies the effective policy non-trivially, invalidating the density computation).
3. **Forward–reverse inconsistency**: The reward is computed at $x_0$, but gradients flow through a different trajectory than the one used during pretraining, creating a gap between the RL and pretraining objectives.

**Idea**: Inject the reward signal into the **forward (noising) process** instead. Use an ODE sampler to generate images and compute rewards, but train the velocity network $v_\theta$ using the standard flow matching MSE loss — reweighted by the reward. No denoising steps are needed for the gradient; no importance ratio is computed; any sampler works.

**Why this works**: The flow matching training loss already sums over all noise levels $t$ for a given clean image $x_0$ (the generated final image, an image or video latent). If we weight each image's contribution by how good it is (reward $r^{(i)} \in [0,1]$, the optimality probability of sample $i$), the model learns to assign higher velocity to the regions of image space that lead to better-rewarded outputs. The forward noising step $x_t = (1-t)x_0 + t\epsilon$ — the straight-line interpolation between clean image and Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ — is the bridge: given a generated image $x_0^{(i)}$, construct noisy versions at all $t$ and train $v_\theta$ on them. The sampler used to produce $x_0^{(i)}$ is irrelevant to the gradient computation.

**Result**: Fully **CFG-free**, DiffusionNFT drives **GenEval 0.24 → 0.98 within ~1k steps** (abstract) vs FlowGRPO's 0.95 in >5k steps *with* CFG (Tab. 1), and is **up to 25× more efficient** than FlowGRPO (3×–25× across four tasks; Figs. 1, 6). At 1.7k iterations it also reaches PickScore 23.80 / HPSv2.1 0.331 / Aesthetic 6.01 / ImageReward 1.49, surpassing CFG-based larger models (SD3.5-L 8B, FLUX.1-dev 12B) from a CFG-free SD3.5-M base (GenEval 0.24).

---

## Implicit Positive and Negative Policies

Rather than explicitly sampling from a reward-weighted distribution, DiffusionNFT defines two **implicit velocity fields** as perturbations around the old policy $v_{\theta_\text{old}}$ (the velocity network at frozen parameters $\theta_\text{old}$ from the previous iteration, which also produced the generations):

$$v_\theta^{+}(x_t, t, c) = (1-\beta)v_{\theta_\text{old}}(x_t, t, c) + \beta v_\theta(x_t, t, c) \quad \text{(positive: aligned with current update)}$$

$$v_\theta^{-}(x_t, t, c) = (1+\beta)v_{\theta_\text{old}}(x_t, t, c) - \beta v_\theta(x_t, t, c) \quad \text{(negative: opposite direction)}$$

Here $v_\theta^{+}$ is the **positive velocity field** — the old field nudged a fraction $\beta$ toward the current update $v_\theta$ — and $v_\theta^{-}$ is the **negative velocity field** — the old field pushed the *same* fraction $\beta$ in the opposite direction; the single scalar $\beta \in (0,1]$ (the mixing/coupling weight) is what builds both implicit fields out of $v_{\theta_\text{old}}$ and $v_\theta$. Note $\tfrac{1}{2}(v_\theta^{+} + v_\theta^{-}) = v_{\theta_\text{old}}$, so the two fields straddle the old policy symmetrically.

These are proxies for:
- $\pi^{+}(x_0|c) \propto r \cdot \pi_{\theta_\text{old}}(x_0|c)$ — a policy biased toward high-reward images
- $\pi^{-}(x_0|c) \propto (1-r) \cdot \pi_{\theta_\text{old}}(x_0|c)$ — a policy biased toward low-reward images

The coupling parameter $\beta$ controls how far the implicit policies deviate from $v_{\theta_\text{old}}$.

**Why contrastive forward training sidesteps the SDE/CFG conflict**: because both fields are matched to the ordinary flow-matching target $u_t^{(i)}$ in the forward direction, the update never references a reverse-step transition density — so there is no stochastic policy to force SDE sampling and no per-step likelihood for CFG to corrupt. The reward-driven gap between $v_\theta^{+}$ and $v_\theta^{-}$ supplies an implicit improvement direction that the paper shows can absorb the role of guidance, which is why DiffusionNFT trains **CFG-free** from a single network.

---

## Training Objective

For each generated image $x_0^{(i)}$ with reward $r^{(i)} \in [0,1]$ (the per-sample optimality probability; see the reference implementation for the raw-reward-to-$[0,1]$ map), construct a forward-noised version. Here $\epsilon^{(i)}$ is a fresh Gaussian draw and $u_t^{(i)}$ is the **target velocity** — the constant flow-matching regression target along the straight path, pointing from noise to the clean image:

$$x_t^{(i)} = (1-t)x_0^{(i)} + t\epsilon^{(i)}, \quad \epsilon^{(i)} \sim \mathcal{N}(0,I), \quad u_t^{(i)} = x_0^{(i)} - \epsilon^{(i)}$$

The **contrastive flow matching loss** combines positive and negative matching weighted by the reward, averaged over the group of $N$ images generated per prompt $c$ (group size $N$). This is a *matching/regression* loss — every term is a flow-matching MSE between a velocity field and the target $u_t^{(i)}$, not a per-step importance ratio — and the RL signal enters only through which field ($v_\theta^{+}$ vs $v_\theta^{-}$) each sample is matched against:

$$\boxed{
\mathcal{L}_\text{NFT}(\theta) = \mathbb{E}_{t,\epsilon}\left[\frac{1}{N}\sum_{i=1}^{N}\left(r^{(i)}\left\Vert{}v_\theta^{+}(x_t^{(i)},t,c) - u_t^{(i)}\right\Vert^2 + (1{-}r^{(i)})\left\Vert{}v_\theta^{-}(x_t^{(i)},t,c) - u_t^{(i)}\right\Vert^2\right)\right]
}$$

Substituting the implicit policy definitions and minimising over $v_\theta$:
- High $r^{(i)}$ ($\approx 1$): loss pushes $v_\theta^{+}$ toward $u_t^{(i)}$, meaning the current update direction $v_\theta - v_{\theta_\text{old}}$ moves toward the target for this rewarded image.
- Low $r^{(i)}$ ($\approx 0$): loss pushes $v_\theta^{-}$ toward $u_t^{(i)}$, meaning the opposite direction $v_{\theta_\text{old}} - v_\theta$ moves toward the target — so $v_\theta$ moves *away* from this poor image.

**Connection to pretraining**: when $r^{(i)} = 0.5$ for all $i$ (uniform reward), $v_\theta^{+}$ and $v_\theta^{-}$ both average to $v_{\theta_\text{old}}$, and the loss reduces to the standard flow matching objective. The RL signal enters only through asymmetric rewards.

---

## Reference Policy Update

Since $v_{\theta_\text{old}}$ appears inside the loss, it must track $v_\theta$ to keep the implicit policies meaningful. A scheduled EMA update:

$$\theta_\text{old} \leftarrow \eta_i\theta_\text{old} + (1-\eta_i)\theta$$

with $\eta_i \to 1$ as training progresses (warm-up toward near-identity). This prevents $v_\theta^{\pm}$ from becoming arbitrary functions far from the current model.

---

## Reference Implementation (VeRL-Omni)

Condensed from [`DiffusionNFTLoss` in `diffusion_algos.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/diffusion/diffusion_algos.py) (`@register_diffusion_loss("diffusion_nft")`). Two stages — (1) the trainer turns raw rewards into a per-sample **optimality probability** `reward_prob ∈ [0,1]` (`prepare_actor_batch`), then (2) the loss reinforces the positive implicit velocity and suppresses the negative one, each as a Tweedie clean-image MSE with adaptive weighting, plus a KL anchor to the reference:

```python
# (1) reward → optimality probability  (DiffusionNFT Sec. 3.3, prepare_actor_batch)
adv = group_normalise(raw_reward, uid)                 # r - mean_group, optionally / std  (GRPO-style)
adv = clamp(adv, -c.adv_clip_max, c.adv_clip_max)      # adv_mode: all | positive_only | binary | ...
reward_prob = clamp(adv / c.adv_clip_max / 2 + 0.5, 0, 1)   # w ∈ [0,1]; w>0.5 ⇒ above-average reward

# (2) forward-process loss
@register_diffusion_loss("diffusion_nft")
def loss_diffusion_nft(forward, old, ref_forward, xt, t, x0, reward_prob, cfg):
    c, beta = cfg.diffusion_loss, cfg.diffusion_loss.mix_beta
    old, ref_forward = old.detach(), ref_forward.detach()
    w   = reward_prob
    pos = beta * forward + (1 - beta) * old             # positive implicit velocity
    neg = (1 + beta) * old - beta * forward             # negative implicit velocity
    x0_pos = xt - t * pos                               # Tweedie → clean image
    x0_neg = xt - t * neg
    wp = abs(x0_pos - x0).mean(dims).clip(min=c.adaptive_weight_min)   # adaptive weights (no grad)
    wn = abs(x0_neg - x0).mean(dims).clip(min=c.adaptive_weight_min)
    pos_loss = ((x0_pos - x0) ** 2 / wp).mean(dims)
    neg_loss = ((x0_neg - x0) ** 2 / wn).mean(dims)
    policy = (w * pos_loss + (1 - w) * neg_loss) / beta * c.adv_clip_max
    kl = ((forward - ref_forward) ** 2).mean()          # KL reg to reference forward field
    return policy.mean() + c.ref_kl_coef * kl
```

---

## Comparison to Reverse-Process Methods

| Aspect | FlowGRPO / DanceGRPO | DiffusionNFT |
|---|---|---|
| Gradient direction | Reverse (denoising) | Forward (noising) |
| SDE required | Yes | **No** — any ODE |
| Importance ratio | Per-step $\rho_t^{(i)}$ | None |
| CFG compatibility | Conflicts with IS ratio | Compatible |
| Objective vs. pretraining | Diverges (noisy IS loss) | Near-identical (flow matching base) |
| Reported speedup | — | ~25× vs. FlowGRPO |

---

## Limitations

- Reward must be normalised to $[0,1]$ (or equivalent) so that positive/negative splits are meaningful; sparse or binary rewards require careful handling.
- Coupling parameter $\beta$ requires tuning — too large destabilises $v_\theta^{\pm}$ as the implicit policies become extreme perturbations.
- No explicit KL regularisation against the reference; relies on EMA and small $\beta$ to prevent mode collapse.
- Theoretical convergence guarantees are weaker than policy gradient methods (the implicit policy construction is heuristic).
