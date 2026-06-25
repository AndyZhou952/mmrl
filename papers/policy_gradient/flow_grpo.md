# FlowGRPO — Flow-GRPO: Training Flow Matching Models via Online RL

| Field | Value |
|---|---|
| **arXiv** | [2505.05470](https://arxiv.org/abs/2505.05470) |
| **Submitted** | 2025-05-08 (v5: 2025-10-27) |
| **Venue** | NeurIPS 2025 |
| **Authors** | Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang |
| **GitHub** | https://github.com/yifan123/flow_grpo |
| **Paradigm** | **Policy Gradient** — per-step Gaussian log-prob required; must use SDE sampler |
| **Cites** | DDPO, GRPO (DeepSeek-R1), PPO, flow matching (Lipman et al.), SD3.5-M, FLUX |
| **Cited by** | MixGRPO, CPS, DiffusionNFT, AWM, GRPO-Guard, DGPO, FlowDPPO |

---

## Context

FlowGRPO is the **first work to apply GRPO to flow matching models**, and it sits at the meeting point of two separate lines of work. This section bridges them — flow matching on one side, GRPO on the other — to show why the combination is non-trivial and what FlowGRPO contributes.

### Flow matching (Lipman et al. 2022; Liu et al. 2022)

Flow matching trains a velocity predictor $v_\theta(x_t, t, c)$ — a network with trainable parameters $\theta$ that, at noisy state $x_t$, flow time $t$, and prompt conditioning $c$, predicts the instantaneous velocity of the transport — that defines a **deterministic ODE** transporting noise to data:

$$dx_t = v_\theta(x_t, t, c) dt, \quad t: 1 \to 0, \quad x_1 \sim \mathcal{N}(0, I)$$

Here $t \in [0,1]$ is continuous flow time with $t=1$ pure noise and $t=0$ clean, $x_1 \sim \mathcal{N}(0, I)$ is the initial Gaussian noise, and $x_t$ is the partially denoised state at time $t$. During inference, an Euler (or higher-order) ODE solver runs this trajectory to produce a clean image $x_0$ (the data sample). This is the backbone of SD3, FLUX, and most state-of-the-art text-to-image/video models. The training loss is velocity matching MSE:

$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t, x_0, x_1}\left[\Vert v_\theta(x_t, t, c) - (x_0 - x_1)\Vert^2\right], \quad x_t = (1-t)x_0 + t x_1$$

The key property: **the ODE trajectory is deterministic** — given initial noise $x_1$, there is exactly one path to $x_0$.

### GRPO for LLMs (DeepSeek-R1, January 2025)

GRPO (Group Relative Policy Optimization) replaces PPO's value network with a group-relative baseline. For a prompt $c$, generate $N_g$ responses $\lbrace{}y^{(i)}\rbrace$ ($N_g$ is the **group size**, the number of samples drawn per prompt), evaluate each with scalar reward $r^{(i)}$, and define the advantage:

$$\hat{A}^{(i)} = \frac{r^{(i)} - \overline{r}}{\mathrm{std}(\lbrace{}r^{(j)}\rbrace) + \delta}$$

where $\overline{r}$ is the group's mean reward, $\mathrm{std}(\cdot)$ its standard deviation, and $\delta > 0$ a small constant for numerical stability. The advantage $\hat{A}^{(i)}$ thus measures how much better sample $i$ scored than its same-prompt peers, in units of within-group spread. The policy gradient uses a PPO-clip objective with per-token importance ratios $\rho_k^{(i)} = \pi_\theta(y_k^{(i)} | y_{<k}^{(i)}, c) / \pi_{\theta_\text{old}}(\cdots)$ — the probability the current policy $\pi_\theta$ assigns to the generated token relative to the frozen sampling policy $\pi_{\theta_\text{old}}$ from the previous iteration. These ratios are tractable because LLM policies are products of categorical softmax distributions — each token has a well-defined probability. GRPO proved highly effective for reasoning in text: cheap to implement (no value network), stable, and sample-efficient.

### Why combining them is non-trivial

GRPO (and PPO generally) requires the importance ratio $\rho = \pi_\theta / \pi_{\theta_\text{old}}$ — the probability ratio of a trajectory under the current vs. old policy. For LLMs, each step contributes a softmax probability, making this straightforward.

For flow matching, the policy is a deterministic ODE step. A deterministic map assigns probability 1 to exactly one output and 0 everywhere else:

$$\pi_\theta(x_{t-\Delta t} \mid x_t, c) = \delta\left(x_{t-\Delta t} - \left[x_t - v_\theta(x_t, t, c)\Delta t\right]\right)$$

This Dirac delta has **no useful density**. There is no $\log \pi_\theta$ to differentiate through, and no ratio $\rho_t$ to compute. Standard policy gradient algorithms cannot be applied as-is.

---

## Problem 1 — ODE has no density; GRPO importance ratio is undefined

**Issue**: Flow matching uses a deterministic ODE. There is no per-step stochastic policy, no density $\pi_\theta(x_{t-\Delta t}|x_t)$, and therefore no importance ratio $\rho_t = \pi_\theta / \pi_{\theta_\text{old}}$. GRPO cannot be applied.

**Idea**: Convert the deterministic ODE to a **stochastically equivalent SDE** that (a) preserves the marginal distribution $p_t(x_t)$ at every $t$, and (b) produces tractable per-step Gaussian log-probabilities.

**Why this works**: By the Fokker-Planck / continuity-equation duality, any probability flow ODE $\dot{x}_t = v(x_t, t)$ has a family of equivalent SDEs:

$$dx_t = \left[v_\theta(x_t, t, c) + \frac{s_t^2}{2}\nabla_{x_t}\log p_t(x_t)\right] dt + s_t dW_t$$

for any diffusion coefficient $s_t > 0$ (a free, typically small hyperparameter controlling how much noise is injected; $s_t \to 0$ recovers the ODE). Here $W_t$ is a standard Wiener process (Brownian motion, the source of the stochasticity), $p_t(x_t)$ is the marginal density of states at time $t$, and $\nabla_{x_t}\log p_t(x_t)$ is its **score** (the gradient of the log-density). The extra drift term $\frac{s_t^2}{2}\nabla \log p_t$ compensates the injected noise, so the marginal $p_t(x_t)$ is identical to the ODE's. FlowGRPO uses this as an engineering device: inject a small, controlled amount of stochasticity during training — enough to define a Gaussian density at each step — while keeping the model's learned distribution intact.

**Result**: Enabling GRPO on flow matching is what unlocks the paper's headline gains on SD3.5-M — GenEval compositional accuracy **63% → 95%** and visual-text-rendering (OCR) accuracy **59% → 92%** (abstract) — with the authors reporting "very little reward hacking."

### Approximating the score

The score $\nabla_{x_t}\log p_t(x_t)$ is unknown, but for the rectified-flow marginal $x_t \mid x_0 \sim \mathcal{N}((1-t)x_0, t^2 I)$ it follows in closed form from the velocity — via **Tweedie's formula**, equivalently the analytic score of that Gaussian marginal:

$$\nabla_{x_t}\log p_t(x_t) \approx \frac{(1-t)\hat{x}_0(x_t, t) - x_t}{t^2} = -\frac{x_t}{t} - \frac{1-t}{t}v_\theta(x_t, t, c), \quad \hat{x}_0 = x_t - t v_\theta(x_t, t, c)$$

where $\hat{x}_0$ is the model's one-shot prediction of the clean image from the current noisy state $x_t$. The velocity predictor $v_\theta$ already encodes this expected clean image $\hat{x}_0$; no additional model is needed.

### Euler-Maruyama discretisation

Discretising the SDE with step size $\Delta t > 0$ (the gap between consecutive sampling times) gives one training step from $t$ to $t - \Delta t$:

$$x_{t-\Delta t} = x_t - \left(v_\theta(x_t, t, c) + \frac{s_t^2}{2}\nabla_{x_t}\log p_t(x_t)\right)\Delta t + s_t\sqrt{\Delta t}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

where $\epsilon_t \sim \mathcal{N}(0, I)$ is fresh Gaussian noise drawn at this step (the drift is the SDE drift above, stepped backward from $t$ to $t-\Delta t$). This step is **Gaussian** with tractable mean and variance:
- Mean: $\mu_\theta(x_t, t, c) = x_t - \left(v_\theta + \dfrac{s_t^2}{2}\nabla_{x_t}\log p_t\right)\Delta t$ — the deterministic part of the update (substitute the score above)
- Variance: $s_t^2\Delta tI$ — set by the diffusion coefficient and step size

The per-step policy density and log-probability are now defined:

$$\pi_\theta(x_{t-\Delta t} \mid x_t, c) = \mathcal{N}\left(x_{t-\Delta t};\ \mu_\theta(x_t, t, c),\ s_t^2\Delta tI\right)$$

$$\log \pi_\theta(x_{t-\Delta t} \mid x_t, c) = -\frac{\Vert{}x_{t-\Delta t} - \mu_\theta(x_t, t, c)\Vert^2}{2s_t^2\Delta t} + \mathrm{const}$$

The importance ratio at step $t$ for sample $i$ — the probability the current policy $\pi_\theta$ assigns to the observed denoising transition $x_t^{(i)} \to x_{t-\Delta t}^{(i)}$, divided by the probability under the frozen rollout policy $\pi_{\theta_\text{old}}$ — is:

$$\rho_t^{(i)} = \frac{\pi_\theta(x_{t-\Delta t}^{(i)} \mid x_t^{(i)}, c)}{\pi_{\theta_\text{old}}(x_{t-\Delta t}^{(i)} \mid x_t^{(i)}, c)} = \exp\left(-\frac{\Vert{}x_{t-\Delta t}^{(i)} - \mu_\theta\Vert^2 - \Vert{}x_{t-\Delta t}^{(i)} - \mu_{\theta_\text{old}}\Vert^2}{2s_t^2\Delta t}\right)$$

This ratio reweights each step's gradient toward transitions the updated policy now favours; $\rho_t^{(i)} = 1$ when the two policies agree. GRPO can now be applied step-by-step, structurally identical to how it operates in LLMs.

---

## Problem 2 — Per-sample reward has high variance

**Issue**: Using the raw reward $r^{(i)}$ as the policy gradient signal is high-variance (REINFORCE-style). Rewards vary across prompts; a good score for one prompt may be mediocre for another. This makes learning unstable.

**Idea**: Adopt GRPO's group-relative advantage. Generate $N_g$ images $\lbrace{}x_0^{(i)}\rbrace_{i=1}^{N_g}$ for the **same** prompt $c$, then normalise rewards within the group:

$$\hat{A}^{(i)} = \frac{r^{(i)} - \overline{r}}{\mathrm{std}(\lbrace{}r^{(j)}\rbrace) + \delta}, \quad \overline{r} = \frac{1}{N_g}\sum_{j=1}^{N_g} r^{(j)}$$

where $r^{(i)} = r(x_0^{(i)}, c)$ is the scalar reward-model score for the $i$-th generated image, $\overline{r}$ is the group mean, and $\delta > 0$ is a small stability constant. The group-relative advantage $\hat{A}^{(i)}$ is shared by every denoising step of trajectory $i$, since the reward is only observed at the final image $x_0^{(i)}$.

**Why this works**: The group mean acts as a prompt-specific baseline, removing variance due to prompt difficulty. Standard-deviation normalisation makes gradients scale-invariant across reward models and prompt batches — the same technique that made GRPO effective in LLM reasoning, transplanted to visual generation. This replaces the raw reward $r^{(i)}$ used in predecessor work (DDPO).

**Result**: Not isolated by an ablation number in the paper; the group baseline is what makes training stable enough to reach the Problem 1 headline gains without the reward collapse seen in raw-reward REINFORCE — see the consolidated abstract results above.

---

## Problem 3 — Full $T$-step gradient computation is expensive

**Issue**: Computing gradients through all $T$ SDE steps ($T$ is the number of denoising steps in the trajectory, $d$ the latent dimension) requires storing all intermediate activations — $O(T \cdot d)$ memory per sample and $O(T)$ sequential compute. With $T=40$ inference steps, this is prohibitive.

**Idea 1 — Denoising reduction**: Use $T_\text{train} \ll T_\text{inf}$ steps during training, where $T_\text{train}$ is the (small) number of denoising steps used to collect RL rollouts and $T_\text{inf}$ the (larger) number used at deployment. Reducing from $T_\text{inf}=40$ to $T_\text{train}=10$ preserves the learning signal (reward is still computed at $x_0$) at $1/4$ the memory cost.

**Idea 2 — FlowGRPO-Fast**: Run a full ODE trajectory (no gradient tracking) to a randomly chosen branch point $t^{\ast}$ (an intermediate flow time at which stochastic sampling is switched on), then take one SDE step $N_g$ times in parallel:

$$x_1 \xrightarrow[\text{no grad}]{\text{ODE}} x_{t^{\ast}} \xrightarrow[\times N_g]{\text{1-step SDE}} \lbrace{}x_{t^{\ast}-\Delta t}^{(i)}\rbrace \xrightarrow[\text{no grad}]{\text{ODE}} \lbrace{}x_0^{(i)}\rbrace$$

**Why Fast works**: The reward is evaluated at $x_0$, not at $t^{\ast}$. The single SDE branch injects enough diversity — through independent noise — for the group advantage $\hat{A}^{(i)}$ to provide a useful gradient signal. Only 1–2 gradient-tracked steps are needed per trajectory, reducing memory cost by $\sim T_\text{train}$× compared to a full rollout.

**Result**: Denoising reduction ($T_\text{inf}{=}40 \to T_\text{train}{=}10$) preserves the same reward signal at ~¼ the memory, and the Fast single-branch variant cuts gradient-tracked steps to 1–2 per trajectory while reaching the gains above; the paper reports the speed/quality trade-off descriptively rather than as a single headline multiplier.

---

## Training Objective

Combining the solutions to all three problems: PPO-clipped GRPO objective summed over all training steps:

$$\boxed{
\mathcal{L}_\text{FlowGRPO}(\theta) = -\mathbb{E}_{c,\lbrace{}x_0^{(i)}\rbrace}\left[\frac{1}{N_g}\sum_{i=1}^{N_g}\frac{1}{T}\sum_{t=1}^{T} \min\left(\rho_t^{(i)}\hat{A}^{(i)},\ \mathrm{clip}\left(\rho_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon\right)\hat{A}^{(i)}\right)\right] + \beta D_\mathrm{KL}(\pi_\theta \Vert \pi_\mathrm{ref})
}$$

where $\rho_t^{(i)}$ is the per-step importance ratio (Problem 1), $\hat{A}^{(i)}$ is the group-relative advantage (Problem 2), and $T = T_\text{train}$ may be reduced per Problem 3. The $\mathrm{clip}(\rho_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon)$ operation bounds the ratio to $[1{-}\epsilon, 1{+}\epsilon]$, with clip width $\epsilon > 0$ (a small constant, e.g. $0.2$) limiting how far each update may move the policy in one step — the standard PPO trust-region heuristic. The final term adds a KL penalty: $D_\mathrm{KL}(\pi_\theta \Vert \pi_\mathrm{ref})$ is the Kullback-Leibler divergence between the current policy $\pi_\theta$ and the frozen pre-RL reference policy $\pi_\mathrm{ref}$, and $\beta \ge 0$ is its weight, anchoring the model near its pretrained behaviour to curb reward hacking.

---

## Reference Implementation (VeRL-Omni)

Condensed from [`diffusion_algos.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/diffusion/diffusion_algos.py) (`@register_diffusion_loss("flow_grpo")`, shared by `"dance_grpo"`). The full `FlowGRPOLoss.compute_loss` reduces to the PPO-clip objective on the per-step log-prob ratio (advantages clamped, optional rollout-correction weights omitted here):

```python
@register_diffusion_loss("flow_grpo")   # also registered for "dance_grpo"
def loss_flow_grpo(old_lp, lp, adv, cfg):
    c = cfg.diffusion_loss
    adv = clamp(adv, -c.adv_clip_max, c.adv_clip_max)
    ratio = exp(lp - old_lp)                                  # ρ_t = π_θ / π_θ_old
    unclipped = -adv * ratio
    clipped   = -adv * clamp(ratio, 1 - c.clip_ratio, 1 + c.clip_ratio)
    return mean(max(unclipped, clipped))                      # PPO-clip
```

---

## Limitations

| Problem | Addressed by |
|---|---|
| SDE noise → image artifacts → misleads reward model | [CPS](cps.md) |
| Importance ratio $\rho_t$ mean $<1$, varying variance → reward hacking | [GRPO-Guard](grpo_guard.md) |
| SDE sampler blocks fast ODE; still expensive | [MixGRPO](mix_grpo.md), [AWM](../direct_preference/awm.md) |
| Pretraining objective $\neq$ GRPO objective | [AWM](../direct_preference/awm.md) |
| Requires SDE → incompatible with ODE-only samplers | [DGPO](../direct_preference/dgpo.md) |
