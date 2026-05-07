# SRPO — Score Regularized Policy Optimization through Diffusion Behavior

> Notation: follows [NOTATION.md](../NOTATION.md). Domain is **offline RL for continuous control** (robotics), not image generation. Actions $a \in \mathbb{R}^{d_a}$; states $s \in \mathbb{R}^{d_s}$.

| Field | Value |
|---|---|
| **arXiv** | [2310.07297](https://arxiv.org/abs/2310.07297) |
| **Submitted** | 2023-10-11 |
| **Venue** | ICLR 2024 |
| **Authors** | Jin-Imok, Ling Pan, Longbo Huang |
| **Domain** | Offline RL — continuous control (D4RL locomotion + manipulation) |
| **Paradigm** | Score-function regularised policy gradient (offline; not image generation) |
| **Cites** | Diffusion policies (Chi et al. 2023), IQL, TD3 |
| **Cited by** | [comparison in offline continuous-control RL]; HunyuanImage 3.0 uses an in-house variant with the same name — see ⚠ note |

> ⚠ **Name collision**: HunyuanImage 3.0 describes an in-house "SRPO" that is a different algorithm (gradient-guided online RL with noise injection for image generation). The academic SRPO here is an offline robotics algorithm.

---

## Motivation

Diffusion policies for offline RL are expressive (multimodal action distributions) but extremely slow at inference — $T=20$–$100$ denoising steps per action. Naïvely distilling into a Gaussian policy loses the multimodal structure. SRPO uses the **score function of the pretrained diffusion behavior model** as a gradient signal to regularise a fast Gaussian policy, achieving the expressiveness of diffusion at the speed of a single forward pass.

---

## Setting

- **Offline dataset** $\mathcal{D} = \{(s, a, s', r)\}$ of transitions.
- **Behavior diffusion model** $\beta(a \mid s)$: a pretrained diffusion model (DDPM-style) over actions, trained on $\mathcal{D}$.
- **Target policy** $\pi_\phi(a \mid s) = \mathcal{N}(a;\, f_\phi(s),\, \sigma^2 I)$: a Gaussian MLP we optimize.
- **Q-function** $Q^\pi(s, a)$: trained offline (e.g., IQL).

At inference the Gaussian policy produces an action in **one forward pass** (no DDPM denoising).

---

## Sampling (inference)

$$a \sim \pi_\phi(\cdot \mid s) = \mathcal{N}(f_\phi(s),\; \sigma^2 I)$$

Single network evaluation — no iterative denoising.

---

## Reward / value calculation

Use a pretrained offline Q-function $Q(s,a)$ (or value function $V(s)$). No online environment interaction.

---

## Training objective

### Objective: balance Q-value with behavior regularisation

$$\max_\phi\; J(\phi) = \mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi_\phi}\!\left[Q(s, a)\right] - \alpha\, \mathbb{E}_{s \sim \mathcal{D}}\!\left[D_\text{KL}(\pi_\phi(\cdot \mid s) \,\|\, \beta(\cdot \mid s))\right]$$

The KL term prevents the policy from deviating from the behavior distribution $\beta$ (offline RL constraint).

### Computing $\nabla_\phi D_\text{KL}(\pi_\phi \| \beta)$ via the score function

The gradient of the KL divergence w.r.t. policy parameters:
$$\nabla_\phi D_\text{KL}(\pi_\phi \| \beta) = \mathbb{E}_{a \sim \pi_\phi}\!\left[\nabla_\phi \log \pi_\phi(a \mid s) \cdot \left(\log \pi_\phi(a \mid s) - \log \beta(a \mid s)\right)\right]$$

Computing $\log \beta(a \mid s)$ is intractable (marginal of diffusion). Instead, differentiate $D_\text{KL}$ through $a$:
$$\nabla_a D_\text{KL}(\pi_\phi \| \beta) = \nabla_a \log \pi_\phi(a \mid s) - \nabla_a \log \beta(a \mid s)$$

The **score function** of the behavior diffusion model $\nabla_a \log \beta(a \mid s)$ is available via the pretrained DDPM score network:
$$\nabla_a \log \beta(a \mid s) \approx -\epsilon_\theta(a_t, t, s) / \sigma_t \Big|_{t \to 0}$$

Using the reparameterisation trick $a = f_\phi(s) + \sigma\,\xi,\; \xi \sim \mathcal{N}(0,I)$:
$$\nabla_\phi D_\text{KL} = \mathbb{E}_\xi\!\left[\left(\nabla_a \log \pi_\phi(a \mid s) - \nabla_a \log \beta(a \mid s)\right)^T \frac{\partial a}{\partial \phi}\right]$$

This is a **reparameterised gradient** — backprop flows through $a \to \phi$ with the score as a fixed signal.

### Full SRPO loss

$$\mathcal{L}_\text{SRPO}(\phi) = -\mathbb{E}_{s,\xi}\!\left[Q(s,\, f_\phi(s) + \sigma\xi)\right] + \alpha\, \mathbb{E}_{s,\xi}\!\left[\left(\frac{f_\phi(s) + \sigma\xi - f_\phi(s)}{\sigma^2} - s_\beta(s,\, f_\phi(s)+\sigma\xi)\right)^T\! \sigma\xi\right]$$

where $s_\beta(s,a) = \nabla_a \log \beta(a \mid s)$ is the score of the behavior distribution (provided by the DDPM score network, queried at a small noise level).

---

## Training algorithm

```
Input: offline dataset D, pretrained DDPM behavior model β with score s_β,
       pretrained Q-function Q, policy init f_φ
Repeat:
  1. Sample state batch {s_i} ~ D
  2. Sample noise: ξ_i ~ N(0,I);  a_i = f_φ(s_i) + σ ξ_i
  3. Q-gradient: ∇_a Q(s_i, a_i)  [backprop through Q]
  4. Score regulariser: s_β(s_i, a_i) via DDPM score network at small t
  5. Policy gradient:
       ∇_φ J ≈ mean_i [ ∇_a Q(s_i,a_i) · ∂a/∂φ
                        - α(a_i/σ² - s_β(s_i,a_i)) · ξ_i ]
  6. φ ← φ + η ∇_φ J
```

No DDPM rollout at training time — only one call to the score network per step.

---

## Relation to image-generation RL

| Concept in SRPO | Analogue in image-generation RL |
|---|---|
| Behavior diffusion model $\beta$ | Pretrained $\pi_\text{ref}$ (base diffusion model) |
| Score function as KL gradient | GRPO-Guard (ratio normalisation informed by diffusion geometry) |
| Gaussian policy $\pi_\phi$ | Implicit in FlowGRPO-Fast / DGPO (fast deterministic sampler) |
| Offline Q-function | Reward model $r(x_0, c)$ |

---

## Limitations

- Specific to continuous-control offline RL; does not apply directly to image generation.
- Gaussian policy assumption limits multimodal expressiveness.
- Score quality depends on the pretrained behavior diffusion model.
- 25× inference speedup over diffusion policies but still requires a pretrained diffusion for training.
