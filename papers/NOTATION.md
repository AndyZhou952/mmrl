# Unified Mathematical Notation

This file defines the symbols used consistently across the algorithm notes in this repository.
The notes are **flow-matching-centric** (rectified flow / velocity prediction), matching the
bulk of current algorithms. DDPM/score-based symbols and the ELBO log-likelihood proxy are
needed by only a couple of methods (DanceGRPO's DDPM SDE branch; DGPO and the Diffusion-DPO
precursor) and are therefore defined **locally** in those files rather than here.

Algorithm files cite back to sections here; when a paper uses a different symbol, the local
mapping is noted at the top of that file.

---

## 1. Data and Conditioning

| Symbol | Meaning |
|---|---|
| $x_0 \in \mathbb{R}^d$ | Clean image (or video latent); sample from data distribution $p_\text{data}$ |
| $\epsilon \sim \mathcal{N}(0, I)$ | Standard Gaussian noise |
| $c$ | Conditioning signal (encoded text prompt) |
| $T$ | Total number of denoising steps (discrete); or $1$ in continuous flow time |
| $t \in \lbrace T, T{-}1, \ldots, 0\rbrace$ | Discrete timestep; **$t=T$ is pure noise, $t=0$ is clean image** throughout this repo |

For continuous flow matching, $t \in [0, 1]$ with the same direction convention: $t=1$ is pure noise, $t=0$ is clean. We write both when it matters.

---

## 2. Noisy States (rectified flow / flow matching)

Linear interpolation path (forward process):

$$x_t = (1-t) x_0 + t \epsilon, \qquad t \in [0,1]$$

| Symbol | Meaning |
|---|---|
| $\lambda_t = 1-t$ | Weight on clean image in interpolation |
| $\mu_t = t$ | Weight on noise in interpolation |
| $u_t = x_0 - \epsilon$ | Target velocity (constant along linear path) |

**Unified shorthand**: $x_t$ denotes the noisy state at step $t$, with $x_T$ (or $x_1$) being pure noise and $x_0$ being the clean image.

> **DDPM / score-based symbols** ($\bar\alpha_t$, $\sigma_t = \sqrt{1-\bar\alpha_t}$, posterior $\tilde\mu_t, \tilde\beta_t$, forward process $x_t = \sqrt{\bar\alpha_t}x_0 + \sigma_t\epsilon$) are used only by **DanceGRPO** (DDPM SDE branch) and **DGPO** / Diffusion-DPO (ELBO). They are defined locally in those files.

---

## 3. Model Parameterisation

| Symbol | Meaning |
|---|---|
| $\theta$ | Current (trainable) model parameters |
| $\theta_\text{old}$ | Frozen parameters from the previous iteration (for importance sampling) |
| $\theta_\text{ref}$ | Frozen reference policy (pre-trained model before RL) |
| $v_\theta(x_t, t, c)$ | Velocity prediction network (flow matching models) |
| $\hat x_0(x_t, t)$ | Predicted clean image via Tweedie's formula: $\hat x_0 = x_t - t v_\theta(x_t,t,c)$ |

(The DDPM noise-prediction net $\epsilon_\theta(x_t,t,c)$, with $\hat x_0 = (x_t - \sigma_t\epsilon_\theta)/\sqrt{\bar\alpha_t}$, is defined locally where used — see §2 note.)

---

## 4. Policy and Probability

The denoising model defines a **policy** $\pi_\theta$ over denoising trajectories.

| Symbol | Meaning |
|---|---|
| $\pi_\theta(x_{t-1} \mid x_t, c)$ | Gaussian transition density under current model |
| $\pi_\text{ref}(x_{t-1} \mid x_t, c)$ | Transition density under reference (frozen) model |
| $\pi_{\theta_\text{old}}(x_{t-1} \mid x_t, c)$ | Transition density under the previous-iteration frozen model |

For flow matching (SDE-converted):

$$\pi_\theta(x_{t-\Delta t} \mid x_t, c) = \mathcal{N}\left(x_{t-\Delta t}; x_t - v_\theta(x_t,t,c)\Delta t + D_t\Delta t, s_t^2\Delta t I\right)$$

where $D_t$ is the SDE drift correction and $s_t$ is the diffusion coefficient (see FlowGRPO).

**Per-step log-likelihood** (used in the GRPO importance ratio):

$$\log \pi_\theta(x_{t-\Delta t} \mid x_t, c) = -\frac{\Vert x_{t-\Delta t} - \mu_\theta(x_t, t, c)\Vert^2}{2 s_t^2 \Delta t} + \text{const}$$

**Importance ratio** (per step):

$$\rho_t^{(i)} = \frac{\pi_\theta(x_{t-\Delta t}^{(i)} \mid x_t^{(i)}, c)}{\pi_{\theta_\text{old}}(x_{t-\Delta t}^{(i)} \mid x_t^{(i)}, c)}$$

---

## 5. Reward and Advantage

| Symbol | Meaning |
|---|---|
| $r(x_0, c) \in \mathbb{R}$ | Scalar reward from a reward model evaluated on final image $x_0$ and prompt $c$ |
| $N$ | Group size: number of images generated per prompt |
| $\lbrace x_0^{(i)}\rbrace_{i=1}^N$ | Group of $N$ images generated from same prompt $c$ |
| $r^{(i)} = r(x_0^{(i)}, c)$ | Reward for sample $i$ in the group |

**Group-relative advantage** (GRPO normalisation):

$$\hat A^{(i)} = \frac{r^{(i)} - \text{mean}\left(\lbrace r^{(j)}\rbrace_{j=1}^N\right)}{\text{std}\left(\lbrace r^{(j)}\rbrace_{j=1}^N\right) + \delta}$$

where $\delta > 0$ is a small stability constant.

---

## 6. Training Objectives

### Flow matching loss (pretraining reference)

$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\Vert v_\theta(x_t, t, c) - u_t\Vert^2\right] = \mathbb{E}_{t, x_0, \epsilon}\left[\Vert v_\theta(x_t, t, c) - (x_0 - \epsilon)\Vert^2\right]$$

This is the pretraining objective that Direct Preference methods (AWM, DiffusionNFT) reuse as the base loss.

### KL regularisation

$$D_\text{KL}(\pi_\theta \Vert \pi_\text{ref}) = \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta(x_{0:T} \mid c)}{\pi_\text{ref}(x_{0:T} \mid c)}\right]$$

In practice approximated per-step and summed.

> The **ELBO log-likelihood proxy** $\log p_\theta(x_0\mid c) \geq -\mathbb{E}_t[\mathcal{L}_\text{FM}] - \text{const}$, which turns an intractable marginal log-ratio into a difference of per-timestep MSE values, is used only by **DGPO** and the **Diffusion-DPO** precursor and is derived locally in those files.

---

## 7. Notation Mapping for Algorithm Notes

Each algorithm note includes a small header block when it deviates from the conventions above.

| Convention | Flow matching (default) | DDPM (defined locally where used) |
|---|---|---|
| Model output | $v_\theta$ (velocity prediction) | $\epsilon_\theta$ (noise prediction) |
| Clean-image prediction | $\hat x_0 = x_t - t v_\theta(x_t,t,c)$ | $\hat x_0 = (x_t - \sigma_t \epsilon_\theta)/\sqrt{\bar\alpha_t}$ |
| Score function | $\nabla_{x_t} \log p_t \approx (\hat x_0 - x_t)/t^2$ | $\nabla_{x_t} \log p_t \approx -\epsilon_\theta / \sigma_t$ |
| Timestep range | $t \in [0,1]$, continuous | $t \in \lbrace 1,\ldots,T\rbrace$, discrete |
| SDE diffusion coeff | $s_t$ (free hyperparameter, small) | $\sigma_t = \sqrt{1-\bar\alpha_t}$ |
