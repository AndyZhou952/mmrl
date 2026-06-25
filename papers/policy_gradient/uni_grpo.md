# UniGRPO — Unified Policy Optimization for Reasoning-Driven Visual Generation

| Field | Value |
|---|---|
| **arXiv** | [2603.23500](https://arxiv.org/abs/2603.23500) |
| **Submitted** | 2026-03-25 |
| **Venue** | — (preprint) |
| **Authors** | Jie Liu, Zilyu Ye, Linxiao Yuan, Shenhan Zhu, Yu Gao, Jie Wu, Kunchang Li, Xionghui Wang, Xiaonan Nie, Weilin Huang, Wanli Ouyang |
| **GitHub** | — |
| **Paradigm** | **Policy Gradient** — SDE-based sampling within window for image policy; autoregressive GRPO for text |
| **Cites** | FlowGRPO (2505.05470), GRPO-Guard (2510.22319), CPS (2509.05952), AWM (2509.25050) |
| **Cited by** | — |

---

## Context

UniGRPO targets **unified multimodal models** — transformers that generate both text reasoning chains and images within a single forward pass (e.g., Transfusion, Show-o style architectures). Prior work applies text RL and image RL as separate loops, missing the opportunity to jointly optimise a shared reward. This file builds on [FlowGRPO](flow_grpo.md) (ODE→SDE for image policy) and [GRPO-Guard](grpo_guard.md) (RatioNorm for ratio stabilisation).

---

## Problem 1 — Text and image RL are fragmented; joint MDP is missing

**Issue**: Existing approaches (e.g., DualGRPO) run two separate RL loops — one for text (standard GRPO over tokens) and one for image (FlowGRPO over denoising steps) — without a shared objective. The reward signal for the image cannot influence the reasoning chain, and vice versa. This means the model cannot learn that a better reasoning chain leads to a better image.

**Idea**: Cast text generation and image generation as a **single unified MDP** with a shared terminal reward $R^{(i)} = R(x_0^{(i)}, y^{(i)}, c)$ — a scalar scoring the complete output of sample $i$: final image $x_0^{(i)}$, reasoning chain $y^{(i)}$, and prompt $c$. The group-relative advantage is computed jointly over the group of $G$ samples drawn from the same prompt:

$$\hat{A}^{(i)} = \frac{R^{(i)} - \overline{R}}{\mathrm{std}(\lbrace{}R^{(j)}\rbrace) + \delta}$$

Here $\hat{A}^{(i)}$ is the **advantage** of sample $i$ (its reward relative to the group, in units of standard deviations), $\overline{R} = \mathrm{mean}(\lbrace{}R^{(j)}\rbrace_{j=1}^G)$ is the group-mean reward, $\mathrm{std}(\cdot)$ is the group standard deviation, and $\delta > 0$ is a small constant for numerical stability. $G$ is the **group size** (number of text+image rollouts per prompt; $G = 24$ in the paper). This advantage flows into both the text gradient (via token-level importance ratios) and the image gradient (via SDE-step importance ratios).

**Why this works**: Sharing the advantage couples text and image optimisation through a common scalar. If a better reasoning chain $y^{(i)}$ produces a better image $x_0^{(i)}$, the shared $\hat{A}^{(i)}$ rewards both the tokens and the denoising steps that led to that outcome — encouraging the model to learn the reasoning-to-image correlation.

**Result**: Joint optimisation beats optimising either component alone — UniGRPO reports **GenEval 0.90** and **Text-Alignment 0.8381**, above FlowGRPO-without-thinking (0.88 / 0.8112) and TextGRPO-alone (0.8078 TA) (Tab. 1).

### Rollout structure

For each prompt $c$:
1. **Text phase**: autoregressively sample $G$ reasoning chains $y^{(i)} \sim \pi_\theta^\text{txt}(\cdot \mid c)$, where $\pi_\theta^\text{txt}$ is the model's text policy (token distribution) and $y^{(i)}$ is the $i$-th sampled chain.
2. **Image phase** (conditioned on $y^{(i)}$): run ODE outside $T_\text{SDE}$, SDE inside $T_\text{SDE}$:

$$\pi_\theta^\text{img}(x_{t_k-\Delta t} \mid x_{t_k}, c, y^{(i)}) = \mathcal{N}\left(x_{t_k-\Delta t};\ \mu_\theta(x_{t_k},t_k,c,y^{(i)}),\ \sigma_{t_k}^2\Delta t I\right), \quad t_k \in T_\text{SDE}$$

where $\pi_\theta^\text{img}$ is the image (denoising) policy, $x_{t_k}$ is the noisy latent at flow time $t_k$ ($t=1$ pure noise, $t=0$ clean), $\Delta t$ the step size, $\mu_\theta(x_{t_k},t_k,c,y^{(i)})$ the Gaussian mean (the SDE Euler step from the predicted velocity), $\sigma_{t_k}$ the per-step SDE noise scale, $I$ the identity covariance, and $T_\text{SDE}$ the **SDE window** — the contiguous subset of denoising steps run stochastically (with gradient); the remaining steps run as a deterministic ODE.

---

## Problem 2 — CFG doubles compute and conflicts with importance ratio computation

**Issue**: Classifier-free guidance (CFG) requires two forward passes per step — one conditioned ($c$, $y^{(i)}$) and one unconditioned ($\varnothing$, the empty condition) — to compute the guided velocity: $v_\text{CFG} = (1{+}w)v_\text{cond} - wv_\text{uncond}$, where $w \geq 0$ is the guidance scale, $v_\text{cond}$ the conditional velocity prediction and $v_\text{uncond}$ the unconditional one. Running CFG through all $T$ steps during training doubles the per-step compute. Moreover, the importance ratio was derived for a single-pass velocity field; the CFG-modified velocity does not correspond to any proper density, breaking the ratio computation.

**Idea**: Apply CFG only during inference (evaluation), and use the standard single-pass (conditional-only) velocity field during training. The SDE steps are run with $v_\theta(\cdot, c, y^{(i)})$ only; guidance is omitted from the training loop.

**Why this works**: The training objective optimises the conditional policy $\pi_\theta^\text{img}$, which is the actual learnable quantity. CFG is a post-hoc inference trick that improves the output without being part of the training distribution. Removing it from training restores the tractability of the importance ratio and halves the forward-pass count per SDE step.

**Result**: Removes the second (unconditional) forward pass per SDE step (~2× less training compute per step) while keeping the importance ratio well-defined.

---

## Problem 3 — Ratio left-shift and KL regularisation in the wrong space

**Issue**: The flow-matching importance ratio is systematically left-shifted (per [GRPO-Guard](grpo_guard.md) analysis), making PPO clip inactive. Additionally, the standard KL penalty — computed in latent space — does not penalise velocity-field drift directly. The model can change $v_\theta$ substantially while the latent KL term remains small, allowing unconstrained policy drift.

**Idea 1 — RatioNorm** (from GRPO-Guard): Standardise the log importance ratio per timestep, restoring zero mean and unit variance. The raw per-step importance ratio is $\rho_{t_k}^{(i)} = \pi_\theta^\text{img}(x_{t_k-\Delta t}^{(i)} \mid x_{t_k}^{(i)},\cdot) / \pi_{\theta_\text{old}}^\text{img}(x_{t_k-\Delta t}^{(i)} \mid x_{t_k}^{(i)},\cdot)$ — the probability the current policy assigns to the sampled denoising step relative to the frozen previous-iteration policy $\pi_{\theta_\text{old}}^\text{img}$. RatioNorm produces a centred surrogate $\tilde{r}_{t_k}^{(i)}$ in its place. Using the Gaussian structure of the SDE step:

$$\log \tilde{r}_{t_k}^{(i)} = -\Delta\mu_{t_k} \cdot \epsilon_{t_k}^{(i)}, \quad \Delta\mu_{t_k} = \mu_\theta(x_{t_k}^{(i)},\cdot) - \mu_{\theta_\text{old}}(x_{t_k}^{(i)},\cdot)$$

where $\Delta\mu_{t_k}$ is the shift in the Gaussian step-mean between the current model $\mu_\theta$ and the frozen previous-iteration model $\mu_{\theta_\text{old}}$, and $\epsilon_{t_k}^{(i)} \sim \mathcal{N}(0,I)$ is the standard Gaussian noise drawn (and stored) at that SDE step during rollout — so the log-ratio is just the projection of the mean-shift onto the realised noise.

**Idea 2 — Velocity-space MSE regularisation**: Replace (or augment) the latent KL with an explicit penalty on the velocity field:

$$\mathcal{L}_\text{MSE}(\theta) = \mathbb{E}_{t_k \in T_\text{SDE},i}\left[\left\Vert{}v_\theta(x_{t_k}^{(i)}, t_k, c, y^{(i)}) - v_{\theta_\text{ref}}(x_{t_k}^{(i)}, t_k, c, y^{(i)})\right\Vert^2\right]$$

where $\mathcal{L}_\text{MSE}$ is the velocity-anchoring loss, $\mathbb{E}_{t_k \in T_\text{SDE},i}[\cdot]$ averages over SDE-window steps and group samples, $v_\theta$ is the current velocity-prediction network and $v_{\theta_\text{ref}}$ the frozen pre-RL reference network ($\theta_\text{ref}$).

**Why this works**: The velocity field $v_\theta$ is the direct output of the model; penalising its $\ell_2$ distance from the reference $v_{\theta_\text{ref}}$ directly constrains the learned dynamics, not just the latent distribution. This is analogous to weight-decay on the output, but semantically meaningful in the flow space.

**Result**: RatioNorm restores clip activation (cf. [GRPO-Guard](grpo_guard.md), which reports the ratio centring and ~20×→2.5× gradient-variance reduction this borrows) and the velocity-space MSE curbs policy drift.

---

## Training Objective

### Text component

Standard GRPO-clip over tokens — $J_\text{Text}$ is the text-side objective (to maximise):

$$J_\text{Text}(\theta) = \frac{1}{G}\sum_{i=1}^G\frac{1}{|y^{(i)}|}\sum_k \left[\min\left(r_{i,k}^\text{txt}\hat{A}^{(i)},\ \mathrm{clip}(r_{i,k}^\text{txt}, 1{-}\epsilon, 1{+}\epsilon)\hat{A}^{(i)}\right) - \beta_\text{txt}D_\text{KL}(\pi_\theta^\text{txt}\Vert\pi_\text{ref}^\text{txt})\right]$$

where $G$ is the group size, $|y^{(i)}|$ the token length of chain $i$ (summed over token index $k$), $r_{i,k}^\text{txt} = \pi_\theta^\text{txt}(y_k^{(i)} \mid \cdot) / \pi_{\theta_\text{old}}^\text{txt}(y_k^{(i)} \mid \cdot)$ the per-token importance ratio, $\hat{A}^{(i)}$ the shared advantage, $\mathrm{clip}(\cdot, 1{-}\epsilon, 1{+}\epsilon)$ the PPO clip with width $\epsilon$ (the surrogate is the elementwise $\min$ of clipped and unclipped terms), and $\beta_\text{txt}$ the weight on the token-level KL penalty $D_\text{KL}(\pi_\theta^\text{txt}\Vert\pi_\text{ref}^\text{txt})$ against the frozen reference text policy $\pi_\text{ref}^\text{txt}$ ($\beta_\text{txt}=0$ in the paper).

### Image component (SDE window, with RatioNorm)

$J_\text{Flow}$ is the image-side (flow) objective over the SDE window:

$$J_\text{Flow}(\theta) = \frac{1}{G}\sum_{i=1}^G\frac{1}{|T_\text{SDE}|}\sum_{t_k \in T_\text{SDE}} \left[\min\left(\tilde{r}_{t_k}^{(i)}\hat{A}^{(i)},\ \mathrm{clip}(\tilde{r}_{t_k}^{(i)}, 1{-}\epsilon, 1{+}\epsilon)\hat{A}^{(i)}\right) - \beta_\text{img}D_\text{KL}(\pi_\theta^\text{img}\Vert\pi_\text{ref}^\text{img})\right]$$

where $|T_\text{SDE}|$ is the number of steps in the SDE window (averaged over step $t_k$), $\tilde{r}_{t_k}^{(i)}$ the RatioNorm-standardised per-step importance ratio (from Problem 3), $\hat{A}^{(i)}$ the same shared advantage as the text term, $\epsilon$ the same clip width, and $\beta_\text{img}$ the weight on the image-side KL against the frozen reference image policy $\pi_\text{ref}^\text{img}$.

### Combined objective

$$\boxed{J_\text{UniGRPO}(\theta) = J_\text{Text}(\theta) + \lambda J_\text{Flow}(\theta) - \gamma\mathcal{L}_\text{MSE}(\theta)}$$

where $\lambda$ balances the flow objective against the text objective ($\lambda = 1$ in the paper) and $\gamma$ is the weight on the velocity-anchoring penalty $\mathcal{L}_\text{MSE}$ (small; the paper uses $\gamma \approx 1.5\times 10^{-5}$). The two policy-gradient terms are maximised while the MSE anchor is subtracted (minimised), keeping the learned velocity field close to the pre-RL reference.

---

## Algorithm

```
Input: unified model θ, reward R(·), prompt dist p_c, group size G, SDE window T_SDE
Initialize: θ_old ← θ,  θ_ref ← θ  (frozen reference)
Repeat:
  1. Sample prompts {c_j}
  2. For each c_j, roll out G trajectories:
       a. Text phase:
            y^(i) ~ π_θ^txt(· | c_j)   [autoregressive, store log probs]
       b. Image phase conditioned on y^(i):
            ODE (no grad):  x_1 → x_{t_max}      # steps outside T_SDE
            SDE (with grad, conditional only — no CFG):
              For t_k ∈ T_SDE:
                Δμ_{t_k} direction via v_θ and Tweedie
                x_{t_k-Δt}^(i) ← μ_θ + σ_{t_k}√Δt · ε,  ε ~ N(0,I)
            ODE (no grad):  tail steps → x_0^(i)
  3. Terminal reward: R^(i) = R(x_0^(i), y^(i), c_j)
  4. Group advantage: Â^(i) = (R^(i) - mean) / std

  5. Text gradient:
       r_k^txt = π_θ^txt(y_k^(i)|·) / π_{θ_old}^txt(y_k^(i)|·)
       J_Text = mean[min(r^txt·Â, clip(r^txt,1-ε,1+ε)·Â) - β_txt·KL]

  6. Image gradient (SDE window, RatioNorm):
       Δμ_{t_k} = μ_θ(x_{t_k}^(i),·) - μ_{θ_old}(x_{t_k}^(i),·)
       log_r̃_{t_k}^(i) = -Δμ_{t_k} · ε_{t_k}^(i)     ← stored noise
       r̃_{t_k}^(i)     = exp(log_r̃_{t_k}^(i))
       J_Flow = mean[min(r̃·Â, clip(r̃,1-ε,1+ε)·Â) - β_img·KL]

  7. Velocity regularisation:
       L_MSE = mean_i,t_k ‖v_θ(x_{t_k}^(i),t_k,c,y^(i)) - v_{θ_ref}(...)‖²

  8. J = J_Text + λ·J_Flow - γ·L_MSE
  9. θ ← θ + η ∇_θ J
  10. θ_old ← θ
```

---

## Comparison to Related Methods

| Aspect | FlowGRPO | GRPO-Guard | DualGRPO | UniGRPO |
|---|---|---|---|---|
| Architecture | Image-only model | Image-only | Separate text + image | Unified transformer |
| Text + image joint MDP | — | — | Separate loops | ✓ |
| Importance ratio | Raw $\rho_t$ | RatioNorm $\hat\rho_t$ | Separate | RatioNorm for image |
| Regularisation | Latent KL | Latent KL | Separate KL | Velocity-space MSE |
| CFG at training | Yes | Yes | Yes | Removed |
| SDE scope | All $T$ steps | All $T$ steps | All $T$ | Window $T_\text{SDE}$ |

---

## Limitations

- Requires a **unified model** architecture (single transformer for text and image); not applicable to encoder-decoder systems or separate text/image models.
- The velocity-space MSE regularisation evaluates $v_{\theta_\text{ref}}$ at each training step — effectively doubling the image forward-pass cost relative to no regularisation.
- Reward must score both text quality (reasoning correctness) and image quality jointly; misspecified rewards risk trading one off against the other.
- RatioNorm is computed per group (size $G$) — approximation degrades for small groups.
