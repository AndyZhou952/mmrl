# Diffusion-DPO — Diffusion Model Alignment Using Direct Preference Optimization

> Notation: follows [NOTATION.md](../NOTATION.md) §6 (ELBO shorthand) and §3 (model). Uses DDPM convention: $\epsilon_\theta$. Preference pairs $(x_0^w, x_0^l)$ share the same prompt $c$.

| Field | Value |
|---|---|
| **arXiv** | [2311.12908](https://arxiv.org/abs/2311.12908) |
| **Submitted** | 2023-11-21 |
| **Venue** | CVPR 2024 |
| **Authors** | Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, Nikhil Naik |
| **Dataset** | Pick-a-Pic (851K crowdsourced pairwise preferences) |
| **Paradigm** | Decoupled — no per-step log-prob; training uses ELBO over stored trajectories |
| **Cites** | DPO (Rafailov et al. 2023), RLHF (Ouyang et al.), DDPM, SDXL |
| **Cited by** | DGPO (extends to online group-level); comparison baseline for FlowGRPO, DanceGRPO |

---

## Motivation

RLHF for diffusion is complex (train reward model → PPO, expensive). DPO for language models directly optimises the policy from preferences but requires computing $\log \pi_\theta(x_0 \mid c)$ — intractable for continuous diffusion models. Diffusion-DPO derives a tractable surrogate using the diffusion ELBO, enabling one-step fine-tuning on stored preference pairs without any online rollouts.

---

## Setting

- **Data**: preference dataset $\mathcal{D} = \{(c, x_0^w, x_0^l)\}$ — each triple has a prompt, a preferred image $x_0^w$, and a dispreferred image $x_0^l$.
- **Model**: DDPM with $\epsilon_\theta$; reference model $\epsilon_{\theta_\text{ref}}$ (frozen copy of pretrained checkpoint).
- **Offline**: no generation during training; all trajectories are fixed.

---

## Sampling (inference)

Standard DDPM (unchanged from pretrained model):
$$x_{t-1} = \mu_\theta(x_t, t, c) + \tilde\beta_t^{1/2}\,\epsilon_t, \quad t = T,\ldots,1$$

Fine-tuning changes $\theta$, which shifts $\mu_\theta$, but the inference protocol is identical.

---

## Reward calculation

No explicit reward model. Instead, the **preference dataset** encodes human judgements: $(x_0^w \succ x_0^l)$ means annotators preferred $x_0^w$. Human-preference labels from Pick-a-Pic, or AI feedback from a VLM.

---

## Training objective

### Step 1 — Standard DPO for LLMs (motivation)

For sequences (LLMs), DPO avoids explicit RL by optimising:
$$\mathcal{L}_\text{DPO}^\text{LLM}(\theta) = -\mathbb{E}_{(c,y^w,y^l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y^w \mid c)}{\pi_\text{ref}(y^w \mid c)} - \beta \log \frac{\pi_\theta(y^l \mid c)}{\pi_\text{ref}(y^l \mid c)}\right)\right]$$

This is a Bradley-Terry classification: the model assigns higher log-probability to the preferred response under a KL-regularised objective. The optimal solution satisfies:
$$\log \frac{\pi^{\ast}(y \mid c)}{\pi_\text{ref}(y \mid c)} = \frac{1}{\beta} r^{\ast}(y, c) - \log Z(c)$$

### Step 2 — Problem: $\log p_\theta(x_0 \mid c)$ is intractable

For diffusion models, $p_\theta(x_0 \mid c) = \int p_\theta(x_{0:T} \mid c)\, dx_{1:T}$ marginalises over all $T$-step denoising paths — a high-dimensional integral with no closed form.

### Step 3 — ELBO lower bound

Introduce latent variables $x_{1:T}$ and use the standard ELBO:
$$\log p_\theta(x_0 \mid c) \geq \mathcal{E}_\theta(x_0, c) \triangleq \mathbb{E}_q\left[\sum_{t=1}^T \log \frac{p_\theta(x_{t-1} \mid x_t, c)}{q(x_{t-1} \mid x_t, x_0)}\right]$$

where $q(x_t \mid x_0)$ is the forward noising process. This gives:
$$\log \frac{p_\theta(x_0^w \mid c)}{p_\text{ref}(x_0^w \mid c)} \approx \mathcal{E}_\theta(x_0^w, c) - \mathcal{E}_{\theta_\text{ref}}(x_0^w, c)$$

### Step 4 — Denoising MSE form

For a DDPM with Gaussian reverse steps, $\log p_\theta(x_{t-1} \mid x_t, c) = -\Vert x_{t-1} - \mu_\theta\Vert^2/(2\tilde\beta_t) + \text{const}$. The ELBO difference simplifies to:

$$\mathcal{E}_\theta(x_0, c) - \mathcal{E}_{\theta_\text{ref}}(x_0, c) = -\mathbb{E}_t\left[\frac{T\,\omega(\lambda_t)}{2}\left(\Vert\epsilon_\theta(x_t,t,c) - \epsilon\Vert^2 - \Vert\epsilon_{\theta_\text{ref}}(x_t,t,c) - \epsilon\Vert^2\right)\right]$$

where $\omega(\lambda_t)$ is a signal-to-noise weighting, $x_t = \sqrt{\bar\alpha_t} x_0 + \sigma_t \epsilon$ is obtained by sampling $\epsilon$ and noising $x_0$.

### Step 5 — Final tractable Diffusion-DPO loss

$$\boxed{
\mathcal{L}_\text{Diff-DPO}(\theta) = -\mathbb{E}_{t,\epsilon}\left[\log \sigma\left(\beta T \omega(\lambda_t)\left[\Delta\mathcal{L}(x_0^w) - \Delta\mathcal{L}(x_0^l)\right]\right)\right]
}$$

where:
$$\Delta\mathcal{L}(x_0) \triangleq \Vert\epsilon_{\theta_\text{ref}}(x_t,t,c) - \epsilon\Vert^2 - \Vert\epsilon_\theta(x_t,t,c) - \epsilon\Vert^2$$

(positive when $\theta$ improves on $\theta_\text{ref}$ for this sample).

**Interpretation**: the loss rewards configurations where the model predicts noise better for the preferred image ($\Delta\mathcal{L}(x_0^w) > 0$) and worse for the dispreferred image ($\Delta\mathcal{L}(x_0^l) < 0$), compared to the reference.

---

## Training algorithm

```
Input: preference dataset D = {(c, x0_w, x0_l)}, frozen θ_ref, β, weighting ω
Repeat for each batch:
  1. Sample (c, x0_w, x0_l) from D
  2. Sample t ~ Uniform{1,...,T}, ε_w, ε_l ~ N(0,I)
  3. Noise both images:
       x_t^w = sqrt(ᾱ_t) x0_w + σ_t ε_w
       x_t^l = sqrt(ᾱ_t) x0_l + σ_t ε_l
  4. Compute denoising losses:
       ΔL(x0_w) = ||ε_θ_ref(x_t^w,t,c) - ε_w||² - ||ε_θ(x_t^w,t,c) - ε_w||²
       ΔL(x0_l) = ||ε_θ_ref(x_t^l,t,c) - ε_l||² - ||ε_θ(x_t^l,t,c) - ε_l||²
  5. L = -log σ(β·T·ω(λ_t)·[ΔL(x0_w) - ΔL(x0_l)])
  6. θ ← θ - η ∇_θ L
```

**Memory**: no trajectories stored; only two forward passes per pair per step. Much lighter than DDPO.

---

## Extension to flow matching

For rectified-flow models (SD3, FLUX), replace the noise prediction MSE with the velocity prediction MSE:
$$\Delta\mathcal{L}_\text{FM}(x_0) = \Vert v_{\theta_\text{ref}}(x_t,t,c) - u_t\Vert^2 - \Vert v_\theta(x_t,t,c) - u_t\Vert^2$$

This is the "Flow-DPO" variant referenced in FlowGRPO ablations.

---

## Limitations

| Limitation | Addressed by |
|---|---|
| **Offline only** — no adaptation to new prompts without new preference data | All online methods (FlowGRPO, DGPO, AWM, …) |
| **ELBO approximation** — timestep weighting $\omega(\lambda_t)$ is a bound, not exact | — |
| **Binary preference** — cannot rank more than two samples | DGPO (extends to group ranking) |
| **No video** — designed for DDPM-style image models | DGPO (video via ELBO), MixGRPO |
