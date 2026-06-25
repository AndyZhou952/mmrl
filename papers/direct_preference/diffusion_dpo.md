# Diffusion-DPO — Diffusion Model Alignment Using Direct Preference Optimization

| Field | Value |
|---|---|
| **arXiv** | [2311.12908](https://arxiv.org/abs/2311.12908) |
| **Submitted** | 2023-11-21 |
| **Venue** | CVPR 2024 |
| **Authors** | Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, Nikhil Naik |
| **GitHub** | https://github.com/SalesforceAIResearch/DiffusionDPO |
| **Paradigm** | **Direct Preference** — ELBO-based pairwise preference loss on final samples; no per-step importance ratio, no SDE (the root of the family) |
| **Cites** | DPO (Rafailov et al. 2023), RLHF/PPO, DDPM, SDXL, Pick-a-Pic |
| **Cited by** | DGPO, DiffusionNFT, AWM, SRPO (as the preference-alignment precursor) |

---

## Context

Diffusion-DPO is the **root of the Direct Preference family**: it was the first to bring [DPO](https://arxiv.org/abs/2305.18290) — the RLHF-free preference method from LLMs — to text-to-image diffusion. Every later direct-preference method in this repo positions against it: [DGPO](dgpo.md) extends its ELBO log-ratio from offline binary pairs to online groups; [AWM](awm.md) and [DiffusionNFT](diffusion_nft.md) replace its preference loss with advantage-weighted / contrastive matching. It is also the workhorse of industry pipelines (the DPO stage in HunyuanImage 3.0, HunyuanVideo, Qwen-Image, Step-Video; see [models.md](../../models.md)). VeRL-Omni ships it as the `dpo` loss and defaults to an **online** variant (sample a group, take best/worst as the pair).

---

## Problem 1 — DPO needs a likelihood; diffusion has no tractable one

**Issue**: DPO aligns a policy to pairwise preferences by a logistic objective in the **log-likelihood ratio** $\log\frac{\pi_\theta(x)}{\pi_\text{ref}(x)}$ between the current model $\pi_\theta$ and a frozen reference model $\pi_\text{ref}$ (the pre-trained checkpoint before alignment). For an LLM that ratio is a product of token softmaxes — immediate. For a diffusion model the marginal likelihood $p_\theta(x_0\mid c)$ of producing image $x_0$ given prompt $c$ is **intractable** (it integrates over all denoising paths $x_{1:T}$), so DPO cannot be applied directly. Before this, the best preference-style tuning for diffusion was just SFT on curated images.

**Idea**: Re-formulate the DPO objective using a **diffusion notion of likelihood** — replace the intractable $\log p_\theta(x_0\mid c)$ with its **ELBO** (evidence lower bound), turning the DPO log-ratio into a *difference of per-timestep denoising MSEs* between the current and reference models. The result is a fully differentiable, simulation-free loss on preference pairs.

**Why this works**: Diffusion-DPO is written in DDPM terms. A clean image $x_0$ is forward-noised at a random timestep $t \in \lbrace 1, \ldots, T\rbrace$ (over $T$ diffusion steps) as $x_t = \sqrt{\bar\alpha_t}x_0 + \sigma_t\epsilon$, with standard Gaussian noise $\epsilon \sim \mathcal{N}(0,I)$, cumulative signal retention $\bar\alpha_t \in (0,1)$, and noise scale $\sigma_t = \sqrt{1-\bar\alpha_t}$; the noise-prediction network is $\epsilon_\theta(x_t,t,c)$, with $\epsilon_\text{ref}$ its frozen pre-alignment copy. Jensen's inequality on the diffusion ELBO then gives $\log p_\theta(x_0\mid c) \geq -\mathbb{E}_{t,\epsilon}[\omega(\lambda_t)\Vert\epsilon_\theta(x_t,t,c) - \epsilon\Vert^2] - \text{const}$, where $\lambda_t=\bar\alpha_t/\sigma_t^2$ is the per-timestep signal-to-noise ratio and $\omega(\lambda_t)$ a non-negative weighting (taken constant in practice). Because the additive constant is the *same* for $\theta$ and $\text{ref}$, it cancels in the log-ratio, and the per-sample DPO log-ratio collapses to a **current-vs-reference error margin** evaluated on a single forward-noised copy of the image:

$$\Delta_\theta(x_0) = \Vert v_\theta(x_t,t,c) - u\Vert^2 - \Vert v_\text{ref}(x_t,t,c) - u\Vert^2, \qquad u = \epsilon - x_0$$

where $v_\theta$ / $v_\text{ref}$ are the current / reference networks and $u$ the regression target ($\epsilon$-noise for DDPM, velocity $u=\epsilon-x_0$ for flow models — the two forms are equivalent up to the $\omega(\lambda_t)$ scaling). The original paper writes this with the DDPM $\epsilon$-prediction MSE, $\Vert\epsilon-\epsilon_\theta(x_t,t,c)\Vert^2 - \Vert\epsilon-\epsilon_\text{ref}(x_t,t,c)\Vert^2$. A negative margin $\Delta_\theta(x_0) < 0$ means the current model denoises (hence "explains") the image *better* than the reference. The Bradley–Terry objective then simply pushes the chosen image's margin below the rejected image's. Because everything is evaluated on the **final** image $x_0$ (forward-noised once per step $t$, not rolled out), there is **no SDE rollout and no per-step importance ratio** — the defining property of the Direct Preference paradigm.

**Result**: Fine-tuning SDXL-1.0 on the **Pick-a-Pic** dataset (851K crowdsourced pairwise preferences) with Diffusion-DPO **significantly outperforms both base SDXL-1.0 and SDXL-1.0 + refiner in human evaluation**, on visual appeal *and* prompt alignment (abstract / paper Fig. 1) — establishing preference optimisation as a stronger alignment route than curated SFT.

---

## Training Objective

Pairwise logistic loss on the current-vs-reference error margin of the chosen ($x_0^w$) over the rejected ($x_0^l$) image, with the expectation taken over preference triples and the shared noising draw $(t,\epsilon)$:

$$\boxed{
\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(c,x_0^w,x_0^l),(t,\epsilon)}\log\sigma\left(-\frac{\beta}{2}\big[\Delta_\theta(x_0^w) - \Delta_\theta(x_0^l)\big]\right)
}$$

where $\sigma(z)=1/(1+e^{-z})$ is the logistic function, $\beta$ the DPO inverse temperature (larger $\beta$ = stiffer regularisation toward $\pi_\text{ref}$ and more sensitivity to the chosen-vs-rejected margin; default $\approx 5000$ in the official code), and $\Delta_\theta$ the ELBO error margin above. The prefactor is the paper's $-\beta T\omega(\lambda_t)$ collapsed to $-\tfrac{\beta}{2}$ under the constant-$\omega$, MSE-mean convention used in implementations (the $T$ is absorbed into $\beta$). The chosen and rejected images are noised with a **shared** $(\epsilon, t)$ so the comparison is apples-to-apples. Note there is **no importance ratio** here: this is a preference loss, and the load-bearing quantities are the margin $\Delta_\theta(x_0^w) - \Delta_\theta(x_0^l)$ and the temperature $\beta$, not a ratio of trajectory likelihoods.

---

## Reference Implementation (VeRL-Omni)

Condensed from [`DPOLoss` in `diffusion_algos.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/diffusion/diffusion_algos.py) (`@register_diffusion_loss("dpo")`). The batch is laid out as adjacent `(chosen, rejected)` pairs (built online by `prepare_actor_batch`: top/bottom reward per prompt). The loss is the per-pair logistic on the current-minus-reference MSE margin:

```python
@register_diffusion_loss("dpo")
def loss_dpo(noise, latent, model_pred, ref_pred, cfg):       # batch = [w0, l0, w1, l1, ...]
    beta   = cfg.diffusion_loss.dpo_beta
    target = noise - latent                                   # flow velocity target u = ε - x0
    model_err = ((model_pred - target) ** 2).mean(non_batch_dims)
    ref_err   = ((ref_pred   - target) ** 2).mean(non_batch_dims)
    w_diff = model_err[0::2] - ref_err[0::2]                   # chosen:  Δ_θ(x0^w)
    l_diff = model_err[1::2] - ref_err[1::2]                   # rejected: Δ_θ(x0^l)
    return -logsigmoid(-0.5 * beta * (w_diff - l_diff)).mean()
```

---

## Limitations

| Problem | Addressed by |
|---|---|
| Offline pairs go stale as the policy improves (fixed dataset) | [DGPO](dgpo.md) (online groups), online-DPO variant |
| Only a binary chosen/rejected signal — no graded/group ranking | [DGPO](dgpo.md) (group Bradley–Terry) |
| ELBO is a loose, timestep-reweighted likelihood proxy | [AWM](awm.md) (advantage-weighted clean-target matching; no ELBO) |
| Requires a frozen reference model in memory | [AWM](awm.md), [SRPO](srpo.md) (no reference) |
