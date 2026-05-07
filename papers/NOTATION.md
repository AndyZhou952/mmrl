# Unified Mathematical Notation

This file defines all symbols used consistently across every algorithm note in this repository.
Algorithm files cite back to sections here; when a paper uses a different symbol, the local mapping is noted at the top of that file.

---

## 1. Data and Conditioning

| Symbol | Meaning |
|---|---|
| $x_0 \in \mathbb{R}^d$ | Clean image (or video latent); sample from data distribution $p_\text{data}$ |
| $\epsilon \sim \mathcal{N}(0, I)$ | Standard Gaussian noise |
| $c$ | Conditioning signal (encoded text prompt) |
| $T$ | Total number of denoising steps (discrete); or $1$ in continuous flow time |
| $t \in \{T, T{-}1, \ldots, 0\}$ | Discrete timestep; **$t=T$ is pure noise, $t=0$ is clean image** throughout this repo |

For continuous flow matching, $t \in [0, 1]$ with the same direction convention: $t=1$ is pure noise, $t=0$ is clean. We write both when it matters.

---

## 2. Noisy States

### DDPM / score-based diffusion

Forward (noising) process:
$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar\alpha_t}\, x_0,\; (1 - \bar\alpha_t) I\right)$$

so $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$ with noise schedule $\{\bar\alpha_t\}_{t=1}^T \in (0,1)$.

| Symbol | Meaning |
|---|---|
| $\bar\alpha_t = \prod_{s=1}^t \alpha_s$ | Cumulative product of retention rates |
| $\sigma_t = \sqrt{1 - \bar\alpha_t}$ | Noise standard deviation at step $t$ |
| $\tilde\mu_t(x_t, x_0)$ | Posterior mean of $q(x_{t-1}\mid x_t, x_0)$ |
| $\tilde\beta_t$ | Posterior variance of $q(x_{t-1}\mid x_t, x_0)$ |

### Rectified flow / flow matching

Linear interpolation path (forward process):
$$x_t = (1-t)\, x_0 + t\, \epsilon, \qquad t \in [0,1]$$

| Symbol | Meaning |
|---|---|
| $\lambda_t = 1-t$ | Weight on clean image in interpolation |
| $\mu_t = t$ | Weight on noise in interpolation |
| $u_t = x_0 - \epsilon$ | Target velocity (constant along linear path) |

**Unified shorthand**: regardless of model family, $x_t$ denotes the noisy state at step $t$, with $x_T$ (or $x_1$) being pure noise and $x_0$ being the clean image.

---

## 3. Model Parameterisation

| Symbol | Meaning |
|---|---|
| $\theta$ | Current (trainable) model parameters |
| $\theta_\text{old}$ | Frozen parameters from the previous iteration (for importance sampling) |
| $\theta_\text{ref}$ | Frozen reference policy (pre-trained model before RL) |
| $\epsilon_\theta(x_t, t, c)$ | Noise prediction network (DDPM models) |
| $v_\theta(x_t, t, c)$ | Velocity prediction network (flow matching models) |
| $\hat x_0(x_t, t)$ | Predicted clean image via Tweedie's formula: $\hat x_0 = \frac{x_t - \sigma_t \epsilon_\theta}{\sqrt{\bar\alpha_t}}$ (DDPM) or $\hat x_0 = x_t - t\, v_\theta$ (flow) |

---

## 4. Policy and Probability

The denoising model defines a **policy** $\pi_\theta$ over denoising trajectories.

| Symbol | Meaning |
|---|---|
| $\pi_\theta(x_{t-1} \mid x_t, c)$ | Gaussian transition density under current model |
| $\pi_\text{ref}(x_{t-1} \mid x_t, c)$ | Transition density under reference (frozen) model |
| $\pi_{\theta_\text{old}}(x_{t-1} \mid x_t, c)$ | Transition density under the previous-iteration frozen model |

For DDPM: $\pi_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t,t,c),\, \tilde\beta_t I)$

For flow matching (SDE-converted): 
$$\pi_\theta(x_{t-\Delta t} \mid x_t, c) = \mathcal{N}\!\left(x_{t-\Delta t};\; x_t - v_\theta(x_t,t,c)\,\Delta t + D_t\,\Delta t,\; s_t^2\,\Delta t\, I\right)$$
where $D_t$ is the SDE drift correction and $s_t$ is the diffusion coefficient (see FlowGRPO).

**Per-step log-likelihood** (used in GRPO importance ratio):
$$\log \pi_\theta(x_{t-1} \mid x_t, c) = -\frac{\|x_{t-1} - \mu_\theta(x_t, t, c)\|^2}{2\tilde\beta_t} + \text{const}$$

**Importance ratio** (per step):
$$\rho_t^{(i)} = \frac{\pi_\theta(x_{t-1}^{(i)} \mid x_t^{(i)}, c)}{\pi_{\theta_\text{old}}(x_{t-1}^{(i)} \mid x_t^{(i)}, c)}$$

---

## 5. Reward and Advantage

| Symbol | Meaning |
|---|---|
| $r(x_0, c) \in \mathbb{R}$ | Scalar reward from a reward model evaluated on final image $x_0$ and prompt $c$ |
| $N$ | Group size: number of images generated per prompt |
| $\{x_0^{(i)}\}_{i=1}^N$ | Group of $N$ images generated from same prompt $c$ |
| $r^{(i)} = r(x_0^{(i)}, c)$ | Reward for sample $i$ in the group |

**Group-relative advantage** (GRPO normalisation):
$$\hat A^{(i)} = \frac{r^{(i)} - \text{mean}\!\left(\{r^{(j)}\}_{j=1}^N\right)}{\text{std}\!\left(\{r^{(j)}\}_{j=1}^N\right) + \delta}$$
where $\delta > 0$ is a small stability constant.

---

## 6. Training Objectives

### Score / flow matching loss (pretraining reference)

DDPM:
$$\mathcal{L}_\text{SM}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon_\theta(x_t, t, c) - \epsilon\|^2\right]$$

Flow matching:
$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|v_\theta(x_t, t, c) - u_t\|^2\right] = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|v_\theta(x_t, t, c) - (x_0 - \epsilon)\|^2\right]$$

### Evidence lower bound (ELBO)

The ELBO bounds the marginal log-likelihood $\log p_\theta(x_0 \mid c)$:
$$\log p_\theta(x_0 \mid c) \geq -\mathbb{E}_t\!\left[\mathcal{L}_\text{SM}(\theta)\right] - \text{const}$$

This is exploited by Diffusion-DPO and DGPO to get tractable log-likelihood proxies.

### Log-ratio shorthand (decoupled methods)

For diffusion models, the log policy-ratio (log importance weight) decomposes over timesteps via the ELBO:
$$\log \frac{\pi_\theta(x_0 \mid c)}{\pi_\text{ref}(x_0 \mid c)} \approx -\mathbb{E}_t\!\left[\mathcal{L}_\text{SM}^\theta(x_0, c) - \mathcal{L}_\text{SM}^\text{ref}(x_0, c)\right]$$

where $\mathcal{L}_\text{SM}^\theta$ is the per-sample matching loss under $\theta$. This turns an intractable marginal into a difference of MSE values.

### KL regularisation

$$D_\text{KL}(\pi_\theta \| \pi_\text{ref}) = \mathbb{E}_{\pi_\theta}\!\left[\log \frac{\pi_\theta(x_{0:T} \mid c)}{\pi_\text{ref}(x_{0:T} \mid c)}\right]$$

In practice approximated per-step and summed, or via the ELBO shorthand above.

---

## 7. MDP for Denoising (DDPO formulation)

| MDP element | Denoising instantiation |
|---|---|
| State $s_t$ | $(c,\, t,\, x_t)$ |
| Action $a_t$ | $x_{t-1}$ (the denoised output at step $t-1$) |
| Transition | Deterministic given $(s_t, a_t)$: next state is $(c, t{-}1, x_{t-1})$ |
| Policy $\pi_\theta(a_t \mid s_t)$ | $p_\theta(x_{t-1} \mid x_t, c)$ |
| Initial state | $s_T = (c, T, x_T)$ with $x_T \sim \mathcal{N}(0,I)$ |
| Reward | $r(x_0, c)$ at $t=0$; zero for all earlier steps |
| Episode | One complete denoising chain $x_T \to x_{T-1} \to \cdots \to x_0$ |

**Policy gradient objective**:
$$J(\theta) = \mathbb{E}_{c \sim p_c,\; \tau \sim \pi_\theta}\!\left[r(x_0, c)\right]$$

**REINFORCE gradient**:
$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(x_{t-1} \mid x_t, c) \cdot r(x_0, c)\right]$$

**PPO-clip per-step loss** (used in DDPO, FlowGRPO, DanceGRPO, MixGRPO):
$$\mathcal{L}_\text{clip}(\theta) = -\mathbb{E}\!\left[\frac{1}{N}\sum_{i=1}^N \frac{1}{T}\sum_{t=1}^T \min\!\left(\rho_t^{(i)} \hat A^{(i)},\; \text{clip}\!\left(\rho_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon\right) \hat A^{(i)}\right)\right]$$

---

## 8. Notation Mapping for Algorithm Notes

Each algorithm note includes a small header block when it deviates:

| Convention | DDPM papers | Flow matching papers |
|---|---|---|
| Model output | $\epsilon_\theta$ (noise prediction) | $v_\theta$ (velocity prediction) |
| Clean-image prediction | $\hat x_0 = (x_t - \sigma_t \epsilon_\theta)/\sqrt{\bar\alpha_t}$ | $\hat x_0 = x_t - t\, v_\theta(x_t,t,c)$ |
| Score function | $\nabla_{x_t} \log p_t \approx -\epsilon_\theta / \sigma_t$ | $\nabla_{x_t} \log p_t \approx (\hat x_0 - x_t)/t^2$ (for linear path) |
| Timestep range | $t \in \{1,\ldots,T\}$, discrete | $t \in [0,1]$, continuous |
| SDE diffusion coeff | $\sigma_t = \sqrt{1-\bar\alpha_t}$ | $s_t$ (free hyperparameter, small) |
