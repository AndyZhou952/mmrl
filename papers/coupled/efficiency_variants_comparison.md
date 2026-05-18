# Efficiency Variants Comparison: FlowGRPO-Fast vs MixGRPO vs DanceGRPO

> Cross-paper comparison of the three concurrent "efficiency" lines. Covers math-level differences,
> code-level differences (original repos), and implementation guidance for
> [verl-omni](https://github.com/verl-project/verl-omni).
>
> Companion papers: [FlowGRPO](flow_grpo.md), [MixGRPO](mix_grpo.md), [DanceGRPO](dance_grpo.md)

---

## 1. At a Glance

| Dimension | FlowGRPO (full) | FlowGRPO-Fast | MixGRPO | MixGRPO-Flash | DanceGRPO |
|---|---|---|---|---|---|
| SDE steps / trajectory | All $T$ | 1 (random $t^{\ast}$) | $w$ (sliding window) | 1 (sliding $w=1$) | $\lceil\tau T\rceil$ (random subset) |
| ODE steps / trajectory | 0 | $T-1$ (Euler) | $T-w$ (Euler) | $T-1$ (DPM-Solver++) | $T-\lceil\tau T\rceil$ (no grad) |
| Window position | N/A | **Random per rollout** | **Slides every $\tau$ training iters** | Same, $w=1$ | **Random per rollout** |
| Loss timestep count | $T$ | 1 | $w$ | 1 | $\lceil\tau T\rceil$ |
| ODE solver quality | — | Standard Euler | Standard Euler | DPM-Solver++ | Standard Euler |
| KL penalty | $\beta D_\text{KL}$ | $\beta D_\text{KL}$ | `kl_coeff` · KL | `kl_coeff` · KL | Optional (backbone-dependent) |
| Loss normalization | $/T$ | $/1$ | $/w$ | $/1$ | $/\lceil\tau T\rceil$ |
| Multi-reward | No | No | Yes (per-reward advantage) | Yes | Yes (VQ+MQ for video) |
| DDPM support | No | No | No | No | **Yes** (SD v1.4) |
| Video support | No | No | No | No | **Yes** (HunyuanVideo, SkyReels) |

---

## 2. Math-Level Comparison

### 2.1 SDE step formula

All methods share the same Gaussian-step form. The differences are in how $\sigma_t$ is defined and where it is applied.

**FlowGRPO / MixGRPO (flow matching, $v_\theta$, $\sigma \in [0,1]$ decreasing):**

$$\mu_\theta = x_t\left(1 + \frac{s_t^2}{2\sigma_t}\Delta\sigma\right) + v_\theta\left(1 + \frac{s_t^2(1-\sigma_t)}{2\sigma_t}\right)\Delta\sigma, \qquad \text{std} = s_t\sqrt{-\Delta\sigma}$$

where $s_t$ is a small fixed hyperparameter (`noise_level` in code, ≈ 0.7–0.8) and $\Delta\sigma = \sigma_{t-1} - \sigma_t < 0$.

**DanceGRPO flow variant** (equivalent but parameterized as $\varepsilon_t = \eta\sqrt{\delta_t}$, $\delta_t = \sigma_t - \sigma_{t-1} > 0$):

$$\mu_\theta = x_t + \Delta\sigma v_\theta + \log\text{-term},\qquad \log\text{-term} = -\tfrac{\varepsilon_t^2}{2}\nabla\log p_t \cdot \Delta\sigma, \qquad \text{std} = \eta\sqrt{\delta_t}$$

with the score $\nabla\log p_t = -(x_t - \hat x_0(1-\sigma_t))/\sigma_t^2$.

**DanceGRPO DDPM variant** ($\epsilon_\theta$, DDIM-style, $\eta \in [0,1]$):

$$\mu_\theta = \sqrt{\bar\alpha_{t-1}}\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\hat\epsilon, \qquad \text{std} = \sigma_t = \eta\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\left(1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}\right)}$$

**Importance ratio** (identical across all methods, only *where* it is applied differs):

$$\rho_t^{(i)} = \frac{\pi_\theta(x_{t-\Delta t}^{(i)}\mid x_t^{(i)},c)}{\pi_{\theta_\text{old}}(x_{t-\Delta t}^{(i)}\mid x_t^{(i)},c)} = \exp\left(-\frac{\Vert x_{t-\Delta t}^{(i)}-\mu_\theta\Vert^2 - \Vert x_{t-\Delta t}^{(i)}-\mu_{\theta_\text{old}}\Vert^2}{2\text{std}^2}\right)$$

ODE steps have $\rho_t \equiv 1$ (no noise → no density → no ratio).

### 2.2 Loss objective

$$\mathcal{L}_\text{FlowGRPO} = -\mathbb{E}\left[\frac{1}{N_g}\sum_i\frac{1}{T}\sum_{t=1}^T \min\left(\rho_t^{(i)}\hat A^{(i)},\text{clip}\left(\rho_t^{(i)},1\pm\epsilon\right)\hat A^{(i)}\right)\right] + \beta D_\text{KL}$$

$$\mathcal{L}_\text{MixGRPO} = -\mathbb{E}\left[\frac{1}{N_g}\sum_i\frac{1}{|\mathcal{W}|}\sum_{t\in\mathcal{W}(l)} \min\left(\rho_t^{(i)}\hat A^{(i)},\text{clip}\left(\rho_t^{(i)},1\pm\epsilon\right)\hat A^{(i)}\right)\right]$$

$$\mathcal{L}_\text{DanceGRPO} = -\mathbb{E}\left[\frac{1}{N_g}\sum_i\frac{1}{\lceil\tau T\rceil}\sum_{t\in\mathcal{T}_\text{sub}} \min\left(\rho_t^{(i)}\hat A^{(i)},\text{clip}\left(\rho_t^{(i)},1\pm\epsilon\right)\hat A^{(i)}\right)\right]$$

**FlowGRPO-Fast** is the limit $w=1$ of MixGRPO with the window selected randomly *per rollout* (not per training iteration), using a standard Euler ODE rather than DPM-Solver++.

### 2.3 The critical subtle difference: FlowGRPO-Fast vs MixGRPO (w=1)

Despite having the same number of gradient steps (1 per trajectory), they differ in:

| | FlowGRPO-Fast | MixGRPO (w=1) |
|---|---|---|
| Window position choice | Random at rollout time → **each trajectory in a batch may have a different $t^{\ast}$** | Fixed at training iteration start → **all trajectories in $\tau$ iters share the same $l$** |
| Coverage over training | Stochastic, covers all $t$ uniformly in expectation | Deterministic sliding, reaches each $t$ exactly once per sweep |
| Gradient signal | Noisy estimate of $\nabla_\theta \mathcal{L}$ for a random $t$ | Systematic sweep: gradient concentrated near window position |
| ODE after branch | Standard Euler (fast, lower quality) | Standard Euler (same as above; DPM++ only in Flash) |

**MixGRPO-Flash** vs **FlowGRPO-Fast**: adds DPM-Solver++ (2nd-order multistep) for all ODE steps *after* the single SDE step, yielding better image quality at the same gradient cost. The 71% training time reduction comes from compressing post-window ODE steps with compression ratio $\tilde r \approx 0.4$.

### 2.4 DanceGRPO timestep subsampling vs FlowGRPO denoising reduction

| | FlowGRPO denoising reduction | DanceGRPO τ-subsampling |
|---|---|---|
| Mechanism | Use fewer total SDE steps: $T_\text{train} \ll T_\text{inf}$ | Use all $T$ SDE steps but compute loss on a random $\tau$-fraction |
| Trajectory type | Shorter SDE trajectory | Full SDE trajectory, random loss mask |
| Gradient coverage | Biased toward last $T_\text{train}$ timesteps | Unbiased over all $T$ timesteps |
| Memory cost | $O(T_\text{train})$ activations | $O(T)$ activations but only $O(\lceil\tau T\rceil)$ backward |

### 2.5 KL penalty formulation

**FlowGRPO**: analytic KL between two Gaussians with same std, computed as MSE of means:

$$D_\text{KL} = \frac{\Vert\mu_\theta - \mu_{\theta_\text{ref}}\Vert^2}{2\text{std}^2}$$

**MixGRPO**: same form but called `kl_coeff * 0.5 * mean((log_π_new - log_π_old)^2)` — an approximation of KL via squared log-ratio.

**DanceGRPO**: KL is often disabled (`beta=0`) for video tasks; used for T2I with CPS SDE type.

---

## 3. Code-Level Comparison

### 3.1 Repository structures

**FlowGRPO** ([yifan123/flow_grpo](https://github.com/yifan123/flow_grpo)):
```
flow_grpo/
├── config/
│   └── grpo.py                          # All hyperparameters (including sde_window_size)
├── flow_grpo/diffusers_patch/
│   ├── sd3_sde_with_logprob.py          # sde_step_with_logprob(): SDE + CPS variants
│   ├── sd3_pipeline_with_logprob.py     # Full T-step SDE pipeline
│   └── sd3_pipeline_with_logprob_fast.py # Fast: random window inside pipeline loop
└── scripts/
    ├── train_sd3.py                     # Full FlowGRPO training loop
    └── train_sd3_fast.py                # Fast variant training loop
```

**MixGRPO** ([Tencent-Hunyuan/MixGRPO](https://github.com/Tencent-Hunyuan/MixGRPO)):
```
MixGRPO/fastvideo/
├── utils/
│   ├── sampling_utils.py                # run_sample_step(), flow_grpo_step(), dpm_step()
│   └── grpo_states.py                   # GRPOTrainingStates: sliding window state
├── models/
│   ├── flux_hf/pipeline_flux.py         # FLUX backbone
│   └── reward_model/                    # HPS-v2.1, PickScore, CLIP, ImageReward
└── train_grpo_flux.py                   # Main training loop; grpo_one_step()
```

**DanceGRPO** ([XueZeyue/DanceGRPO](https://github.com/XueZeyue/DanceGRPO)):
```
DanceGRPO/fastvideo/
├── models/
│   ├── stable_diffusion/
│   │   └── ddim_with_logprob.py         # DDPM reverse SDE: ddim_step_with_logprob()
│   └── hunyuan/diffusion/schedulers/
│       └── scheduling_flow_match_discrete.py  # Flow matching scheduler
├── utils/load.py                        # Model dispatcher: load_transformer(), load_vae()
└── train_grpo_{flux,sd,hunyuan,skyreels_i2v,wan_2_1}.py  # One script per backbone
```

### 3.2 ODE/SDE switching mechanism

**FlowGRPO original** — `noise_level` controls stochasticity:
```python
# sd3_pipeline_with_logprob_fast.py
if i < sde_window[0] or i >= sde_window[1]:
    cur_noise_level = 0      # ODE: deterministic Euler step
else:
    cur_noise_level = noise_level  # SDE: stochastic with logprob
```
Window is chosen randomly *inside* the pipeline each forward pass:
```python
start = random.randint(sde_window_range[0], sde_window_range[1] - sde_window_size)
sde_window = (start, start + sde_window_size)
```

**MixGRPO / DanceGRPO** — `determistic[i]` boolean array:
```python
# train_grpo_flux.py (MixGRPO)
if args.training_strategy == "part":   # MixGRPO
    determistic = [True] * sample_steps
    for i in timesteps_train:          # window indices from GRPOTrainingStates
        determistic[i] = False         # SDE inside window
elif args.training_strategy == "all":  # DanceGRPO
    determistic = [False] * sample_steps  # all SDE

# sampling_utils.py: run_sample_step() reads determistic[i] at each step
z, _, log_prob, _ = flow_grpo_step(
    model_output=pred, latents=z, sigmas=sigma_schedule,
    index=i, determistic=determistic[i],  # ← key flag
    sde_type=args.sde_type,
)
```

**Sliding window state** (`grpo_states.py`, MixGRPO-specific):
```python
class GRPOTrainingStates:
    cur_timestep: int          # l — current window start
    group_size: int            # w
    iters_per_group: int       # τ
    prog_overlap_step: int     # s

    def get_current_timesteps(self) -> List[int]:
        return list(range(self.cur_timestep,
                          min(self.cur_timestep + self.group_size, self.max_timesteps)))

    def update_iteration(self):
        # Every τ iters: l ← min(l + s, T - w)
        if self.iter_count % self.iters_per_group == 0:
            self.cur_timestep = min(self.cur_timestep + self.prog_overlap_step,
                                    self.max_timesteps - self.group_size)
```
This object is **created once and mutated throughout training** — it is the key missing piece from verl-omni.

### 3.3 SDE step formulas in code

All three repos implement the same Gaussian transition but with slightly different naming:

```python
# MixGRPO / DanceGRPO flow variant: flow_grpo_step() in sampling_utils.py
sigma      = sigmas[index]           # σ_t
sigma_prev = sigmas[index + 1]       # σ_{t-1}
dt         = sigma_prev - sigma      # Δσ < 0

std_dev_t = sqrt(sigma / (1 - sigma)) * eta   # s_t (paper's noise_level)
prev_sample_mean = (
    latents * (1 + std_dev_t**2 / (2*sigma) * dt) +
    model_output * (1 + std_dev_t**2 * (1-sigma) / (2*sigma)) * dt
)
prev_sample = prev_sample_mean + std_dev_t * sqrt(-dt) * randn_like(latents)  # SDE
# or: prev_sample = latents + dt * model_output                                # ODE

log_prob = (
    -((prev_sample.detach() - prev_sample_mean)**2)
    / (2 * (std_dev_t * sqrt(-dt))**2)
    - log(std_dev_t * sqrt(-dt))
    - log(sqrt(2π))
)

# DanceGRPO flow variant: flux_step() uses slightly different reparameterization
# std_dev_t = eta * sqrt(delta_t)  where delta_t = sigma - sigma_prev > 0
# score_estimate = -(x_t - x_0_hat*(1-sigma)) / sigma**2
# log_term = -0.5 * eta**2 * score_estimate * dsigma
```

**CPS variant** (available in both FlowGRPO and MixGRPO via `--sde_type cps`):
```python
std_dev_t = sigma_prev * sin(noise_level * π/2)   # different noise scale
prev_sample_mean = pred_original_sample * (1 - sigma_prev) + \
                   noise_estimate * sqrt(sigma_prev**2 - std_dev_t**2)
```

**DanceGRPO DDPM variant** (`ddim_step_with_logprob.py`):
```python
alpha_prod_t      = alphas_cumprod[timestep]
alpha_prod_t_prev = alphas_cumprod[prev_timestep]
beta_prod_t       = 1 - alpha_prod_t
variance          = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
std_dev_t         = eta * sqrt(variance)
pred_original_sample = (sample - sqrt(beta_prod_t) * model_output) / sqrt(alpha_prod_t)
pred_sample_direction = sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * pred_epsilon
prev_sample_mean  = sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
```

### 3.4 Importance ratio and loss (where the code diverges)

**FlowGRPO original** (loops over all timesteps):
```python
for j in range(num_train_timesteps):           # j = 0 .. T-1
    log_prob  = compute_log_prob(transformer, pipeline, sample, j, ...)
    ratio     = exp(log_prob - sample["log_probs"][:, j])
    policy_loss += ppo_clip(ratio, advantages[:, j], clip_range)
    kl_loss     += mse_of_means(...)
```

**MixGRPO** (loops over window only, normalized by `len(train_timesteps)`):
```python
for _ in train_timesteps:                      # only W(l) indices
    new_log_probs = grpo_one_step(...)
    ratio = exp(new_log_probs - sample["log_probs"][:, _])
    policy_loss += ppo_clip(ratio, advantages, clip_range) \
                   / (gradient_accumulation_steps * len(train_timesteps))
    kl_loss     += 0.5 * mean((new_log_probs - sample["log_probs"][:, _])**2) \
                   / (gradient_accumulation_steps * len(train_timesteps))
    loss.backward()
```

**DanceGRPO** (random permutation → first `train_timesteps` indices):
```python
perms = [randperm(len(timesteps)) for _ in range(batch)]
# shuffle latents / log_probs by perms
train_timesteps = int(len(timesteps) * args.timestep_fraction)
for _ in range(train_timesteps):               # first τ·T shuffled steps
    new_log_probs = grpo_one_step(...)
    ratio = exp(new_log_probs - sample["log_probs"][:, _])
    policy_loss += ppo_clip(ratio, advantages, clip_range) \
                   / (gradient_accumulation_steps * train_timesteps)
    loss.backward()
```

### 3.5 Advantage computation

All methods use the same group-relative formula; the differences are in aggregation:

```python
# GRPO standard (all three methods)
advantage[i] = (reward[i] - group_mean) / (group_std + 1e-8)

# FlowGRPO: per-prompt EMA tracker (PerPromptStatTracker.update())
#   — tracks running mean/std per prompt string across rollouts
# MixGRPO: per-prompt within-batch normalization
#   — with optional trimmed_ratio to remove outlier rewards
# DanceGRPO Hunyuan: separate advantage per reward model, then linear combination
#   total_score = vq_coeff * vq_adv + mq_coeff * mq_adv
```

### 3.6 Reference model handling

**FlowGRPO**: EMA of training weights (`ema.py`):
```python
ema_param = decay * ema_param + (1 - decay) * param   # decay = 0.9999
# Reference = disable LoRA adapter → base model weights
with transformer.module.disable_adapter():
    _, _, ref_mean, _ = compute_log_prob(...)
```

**MixGRPO / DanceGRPO**: Implicit frozen reference.
- Log probabilities are precomputed during the rollout phase (`sample_reference_model()`) **before** any gradient steps.
- Used throughout the epoch as `sample["log_probs"]`.
- No explicit "θ_old ← θ" update inside the epoch; the reference shifts only at epoch boundaries.

### 3.7 DPM-Solver++ integration (MixGRPO-Flash only)

```python
# sampling_utils.py: build compressed post-window ODE schedule
num_post_steps = int((T - last_sde_index) * dpm_post_compress_ratio)  # r̃·remaining
post_sigma_schedule = linspace(sigma_at_last_sde, 0, num_post_steps)

# dpm_solver_first_order_update():
lambda_t = log(alpha_t) - log(sigma_t)
lambda_s = log(alpha_s) - log(sigma_s)
h = lambda_t - lambda_s
x_t = (sigma_t/sigma_s) * sample - alpha_t * (exp(-h) - 1) * model_output
# 2nd-order multistep: uses prev_model_output from t-1 step for correction
```

This is what gives MixGRPO-Flash its 71% training time reduction — fewer ODE steps with higher order means fewer transformer forward passes outside the window.

---

## 4. verl-omni Status and Implementation Guidance

### 4.1 Current state of verl-omni

verl-omni currently implements **FlowGRPO-Fast** (Qwen-Image backbone only):

| Component | File | Status |
|---|---|---|
| SDE scheduler | `pipelines/schedulers/flow_match_sde.py` | FlowGRPO SDE + CPS variants ✅ |
| Random window | `sde_window_range` + `sde_window_size` in config | FlowGRPO-Fast ✅ |
| Loss / ratio | `trainer/diffusion/diffusion_algos.py` | PPO-clip + optional KL ✅ |
| Group advantage | `compute_flow_grpo_outcome_advantage()` | Standard GRPO ✅ |
| DiffusionModelBase | `pipelines/model_base.py` | Abstraction layer ✅ |
| Sliding window state | — | **Missing for MixGRPO** ❌ |
| DDPM scheduler | — | **Missing for DanceGRPO** ❌ |
| DPM-Solver++ | — | **Missing for Flash** ❌ |
| Multi-reward | — | **Missing** ❌ |
| Video backbones | — | Wan2.2 (in progress), Hunyuan planned |

### 4.2 Implementing MixGRPO in verl-omni

**Step 1 — Window state tracker** (new file: `trainer/diffusion/mixgrpo_window.py`):
```python
class SlidingWindowState:
    """Persistent state tracking the current SDE window position."""
    def __init__(self, window_size: int, stride: int, shift_interval: int, max_steps: int):
        self.window_start = 0        # l
        self.window_size  = window_size  # w
        self.stride       = stride       # s
        self.shift_interval = shift_interval  # τ
        self.max_steps    = max_steps
        self._iter_count  = 0

    def get_window(self) -> List[int]:
        end = min(self.window_start + self.window_size, self.max_steps)
        return list(range(self.window_start, end))

    def step(self):
        """Call once per training iteration."""
        self._iter_count += 1
        if self._iter_count % self.shift_interval == 0:
            self.window_start = min(self.window_start + self.stride,
                                    self.max_steps - self.window_size)
```

**Step 2 — Scheduler extension**: The existing `FlowMatchSDEDiscreteScheduler` already handles per-step SDE/ODE switching via `noise_level=0`. Expose a `step_is_sde: bool` flag instead:
```python
def step(self, model_output, timestep, sample, step_is_sde: bool = True, ...):
    if not step_is_sde:
        # ODE: Euler step, no log_prob needed
        return sample + (sigma_prev - sigma) * model_output, None
    # existing SDE path ...
```

**Step 3 — Rollout path**: In the rollout worker (`agent_loop/`), pass the current window from `SlidingWindowState` to the pipeline:
```python
window = window_state.get_window()
determistic = [i not in window for i in range(T)]
# run pipeline with determistic flags
```
The `forward_and_sample_previous_step()` abstraction in `DiffusionModelBase` needs a `step_is_sde` parameter forwarded per step.

**Step 4 — Training loss**: In `diffusion_algos.py`, restrict the loss loop to window steps:
```python
def compute_diffusion_loss_mixgrpo(data, window_indices, ...):
    total_loss = 0
    for t in window_indices:             # only W(l) steps
        ratio     = exp(log_probs[:, t] - old_log_probs[:, t])
        total_loss += ppo_clip(ratio, advantages, clip_range)
    return total_loss / len(window_indices)
```

**Step 5 — Trainer loop**: Advance window state after each update:
```python
# ray_diffusion_trainer.py
for batch in dataloader:
    rollout_data = rollout_worker.generate(batch, window=window_state.get_window())
    loss = compute_diffusion_loss_mixgrpo(rollout_data, window_state.get_window())
    actor.update(loss)
    window_state.step()                  # ← advance window
```

**Config additions** (`diffusion_trainer.yaml`):
```yaml
algorithm:
  algo: mixgrpo          # new value (currently: flowgrpo)
  window_size: 4         # w
  window_stride: 1       # s
  window_shift_interval: 25  # τ
  # For Flash variant:
  use_dpm_solver: false
  dpm_compress_ratio: 0.4
```

**Coupling to untangle first**:
- `sde_window_range` / `sde_window_size` in the rollout config are per-rollout random choices (FlowGRPO-Fast semantics). For MixGRPO these must become deterministic, driven by the persistent window state. Remove or deprecate the random window path when `algo=mixgrpo`.
- The `vllm_omni_rollout_adapter.py` currently picks the window inside the pipeline; move this decision up to the trainer so the window can be shared from the state tracker.

### 4.3 Implementing DanceGRPO in verl-omni

**Step 1 — DDPM scheduler** (new file: `pipelines/schedulers/ddpm_sde.py`):
Adapt `ddim_step_with_logprob()` from DanceGRPO's stable diffusion code:
```python
class DDPMSDEScheduler(DDIMScheduler):
    def step_with_logprob(self, model_output, timestep, sample, eta=1.0):
        alpha_prod_t      = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[max(timestep - self.step_size, 0)]
        variance          = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * \
                            (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t         = eta * variance**0.5
        pred_x0           = (sample - (1-alpha_prod_t)**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir          = (1 - alpha_prod_t_prev - std_dev_t**2)**0.5 * model_output
        mean              = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        noise             = randn_like(sample)
        prev_sample       = mean + std_dev_t * noise
        log_prob = (-((prev_sample.detach() - mean)**2) / (2 * std_dev_t**2)
                    - log(std_dev_t) - log(sqrt(2π)))
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, log_prob
```

**Step 2 — Timestep subsampling** in the training loss:
```python
def compute_diffusion_loss_dancegrpo(data, timestep_fraction=0.6, ...):
    T = data["log_probs"].shape[1]
    # Random shuffle of timestep indices per sample
    perm = torch.randperm(T)
    n_train = math.ceil(T * timestep_fraction)
    train_indices = perm[:n_train]
    # Loss over random subset
    total_loss = 0
    for t in train_indices:
        ratio = exp(log_probs[:, t] - old_log_probs[:, t])
        total_loss += ppo_clip(ratio, advantages, clip_range)
    return total_loss / n_train
```

**Step 3 — Model adapter for SD**: Implement `DiffusionModelBase` subclass for Stable Diffusion using the DDPM scheduler above. Register via `@DiffusionModelBase.register("stable-diffusion-v1-4")`.

**Step 4 — Multi-reward advantage**: Add reward combination logic in `diffusion_algos.py`:
```python
def compute_flow_grpo_outcome_advantage_multireward(rewards_dict, reward_weights, num_gen):
    """Compute per-reward advantage and linear-combine."""
    combined = torch.zeros(total_samples)
    for name, rewards in rewards_dict.items():
        for i in range(n_prompts):
            g = rewards[i*num_gen:(i+1)*num_gen]
            adv = (g - g.mean()) / (g.std() + 1e-8)
            combined[i*num_gen:(i+1)*num_gen] += reward_weights[name] * adv
    return combined
```

**Config additions**:
```yaml
algorithm:
  algo: dancegrpo
  timestep_fraction: 0.6    # τ
  rewards:
    - name: hps
      weight: 0.5
    - name: clip
      weight: 0.5
```

### 4.4 Implementing MixGRPO-Flash (DPM-Solver++ ODE)

This is optional and can be added after basic MixGRPO works. The key additions:

1. **DPM-Solver++ state**: Multi-step method needs the previous model output cached during the ODE phase.
2. **Compressed schedule**: After the SDE window ends, rebuild the sigma schedule with fewer steps using `linspace` + `sd3_time_shift`.
3. **Step function**: Implement `dpm_solver_second_order_update()` following MixGRPO's `dpm_step()`.

Recommended to wire this as a flag: `use_dpm_solver_flash: true` under `algorithm`.

### 4.5 Priority order for verl-omni

1. **MixGRPO** (highest impact, HunyuanImage-3.0 production use case):
   - `SlidingWindowState` class
   - Make window position trainer-controlled (not pipeline-internal)
   - Window-restricted loss in `diffusion_algos.py`

2. **DanceGRPO timestep subsampling** (for video / multi-backbone):
   - Timestep fraction loss mask
   - Multi-reward advantage
   - (DDPM scheduler is lower priority as Wan2.2 is flow-matching)

3. **MixGRPO-Flash** (nice-to-have, reduces training time ~70%):
   - DPM-Solver++ state + compressed schedule
   - Only needed once basic MixGRPO is working

4. **DDPM support** (SD v1.4 / v1.5):
   - New scheduler + model adapter
   - Lowest priority unless a DDPM backbone is planned

---

## 5. Quick Reference: Key Differences Summary

### Math

| | Where SDE is applied | Window selection | Loss denominator |
|---|---|---|---|
| FlowGRPO | All $T$ steps | N/A | $T$ |
| FlowGRPO-Fast | 1 step: random per rollout | Per-rollout random $t^{\ast}$ | 1 |
| MixGRPO | $w$ steps: sliding window | Per-$\tau$-iters sliding | $w$ |
| MixGRPO-Flash | 1 step: sliding window | Per-$\tau$-iters sliding | 1 |
| DanceGRPO | $\lceil\tau T\rceil$ steps: random per rollout | Per-rollout random $\tau$-fraction | $\lceil\tau T\rceil$ |

### Code

| | Window control | ODE/SDE flag | SDE formula location |
|---|---|---|---|
| FlowGRPO | `sde_window` chosen in pipeline | `noise_level = 0` | `sde_step_with_logprob()` |
| MixGRPO | `GRPOTrainingStates.cur_timestep` | `determistic[i]` array | `flow_grpo_step()` |
| DanceGRPO | `randperm + timestep_fraction` | `sde_solver: bool` arg | `flux_step()` / `ddim_step_with_logprob()` |
| verl-omni | `sde_window_range` (random, rollout-time) | `noise_level` in scheduler | `FlowMatchSDEDiscreteScheduler.step()` |

### verl-omni gaps

| Feature | Gap | Estimated effort |
|---|---|---|
| Sliding window state | New `SlidingWindowState` class + trainer integration | Small |
| Trainer-controlled window (vs rollout-internal) | Refactor `vllm_omni_rollout_adapter.py` | Medium |
| Window-restricted loss | New branch in `diffusion_algos.py` | Small |
| Timestep subsampling | New branch in loss + random permutation | Small |
| Multi-reward advantage | New function in `diffusion_algos.py` | Small |
| DDPM scheduler | New scheduler class + model adapter | Medium |
| DPM-Solver++ | New solver + compressed schedule | Medium |
