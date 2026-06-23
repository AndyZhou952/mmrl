# Prerequisite — Flow Matching (and just enough diffusion)

> Notation: follows [NOTATION.md](../NOTATION.md). Scoped to what diffusion RL needs. We assume you are comfortable reading a formula; we skip the proofs and keep the intuition.

Almost every algorithm in this repo fine-tunes a **flow matching** image/video model. You need three things: what the model *is*, how it is *trained and sampled*, and the **one structural fact the entire field is built around** — the sampler is a deterministic ODE that has *no per-step probability*, which is exactly what RL needs.

---

## 1. The goal, in one picture

A generator turns noise into data. Picture two cities: **Noise-land** (samples of $\epsilon \sim \mathcal{N}(0,I)$) and **Data-land** (real images/videos, as latents $x_0$). We want a reliable route from any point in Noise-land to a good point in Data-land, conditioned on the prompt $c$.

Flow matching learns that route as a **velocity field** $v_\theta(x_t, t, c)$ — read it as *driving directions*: "if you are at position $x_t$ at time $t$, drive in this direction." Following the directions from $t=1$ (pure noise) down to $t=0$ (clean image) traces one trajectory from noise to data. (Throughout this repo, $t=1$ is noise and $t=0$ is clean.)

## 2. The training path: a straight line

Flow matching picks the simplest possible route during training — a **straight line** between a data point and a noise point:

$$x_t = (1-t)x_0 + t\epsilon, \qquad t \in [0,1]$$

The direction you must drive along that straight line is constant — it is just the displacement from noise to data:

$$u_t = x_0 - \epsilon$$

So the **training target is the straight-line displacement**. That is the whole idea of *rectified* flow: prefer straight routes.

## 3. Training: regress the directions (velocity matching)

Train the network to predict that displacement, by plain MSE regression:

$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\Vert v_\theta(x_t, t, c) - (x_0 - \epsilon)\Vert^2\right]$$

This is the **pretraining objective**. Keep it in mind: the Direct Preference family (AWM, DiffusionNFT) reuses *this exact loss* and just re-weights it by a reward — that is the whole trick, no new machinery.

From $v_\theta$ you can also read off a one-step guess of the clean image (**Tweedie's formula**), which several methods use to score partial generations:

$$\hat x_0 = x_t - t v_\theta(x_t, t, c)$$

## 4. Sampling: follow the directions (integrate an ODE)

To generate, start at noise $x_1$ and integrate the **probability-flow ODE** from $t=1$ to $t=0$ with an Euler (or higher-order) step:

$$dx_t = v_\theta(x_t, t, c) dt \quad\Rightarrow\quad x_{t-\Delta t} \approx x_t - v_\theta(x_t, t, c)\Delta t$$

Why straight paths matter: Euler takes *straight-line* steps. On a straight route you can take a few big steps and still arrive; on a curvy route you need many tiny steps or you overshoot. As the rectified-flow folks put it, **"curvature is the enemy of speed."** Straighter trajectories ⇒ fewer denoising steps ⇒ cheaper generation (the basis of fast samplers like DPM-Solver++).

## 5. The crux: the ODE is a *train on rails*

Here is the fact the whole survey turns on. The sampling step in §4 is **deterministic**: given $x_t$, the next state $x_{t-\Delta t}$ is fixed. The "policy" at each step is a spike — it puts probability 1 on a single output and 0 everywhere else:

$$\pi_\theta(x_{t-\Delta t}\mid x_t, c) = \delta\big(x_{t-\Delta t} - [x_t - v_\theta\Delta t]\big)$$

A deterministic generator is a **train on rails**: it always goes to the same place. RL algorithms (PPO/GRPO) need two things the rails do not give you:

1. **Exploration** — try slightly different outputs to discover which are better.
2. **A probability for each move** — to form the importance ratio $\rho = \pi_\theta/\pi_{\theta_\text{old}}$ that policy gradients optimise.

A spike has no usable $\log \pi_\theta$, so there is nothing to differentiate or compare. This is *the* obstacle every algorithm in this repo answers.

## 6. The two escapes (and how they map to VeRL-Omni)

- **Policy Gradient family** — *add steering jitter*. Convert the deterministic ODE into a **stochastic SDE** that has the *same marginal distribution at every $t$* but injects a little controlled noise, turning each step into a Gaussian you can sample from and score. Now the train can wander a bit (exploration) and each wiggle has a probability (density). This is exactly [FlowGRPO](../policy_gradient/flow_grpo.md). In **VeRL-Omni** this is the scheduler flag `sde_type` (`sde` = FlowGRPO, `cps` = the coefficients-preserving variant from [CPS](../policy_gradient/cps.md)).
- **Direct Preference family** — *don't bother with per-step probabilities at all*. Generate with any fast ODE sampler, then define the loss on the **final samples** (a preference/MSE objective). No density, no SDE — see [AWM](../direct_preference/awm.md), [DiffusionNFT](../direct_preference/diffusion_nft.md), [DGPO](../direct_preference/dgpo.md).

That single fork — *make the sampler stochastic to get a density*, or *avoid needing one* — is the backbone of the [taxonomy](../INDEX.md).

## 7. DDPM in one paragraph (only where it appears)

A couple of methods touch the older **DDPM** parameterisation: a noise-prediction network $\epsilon_\theta$ and a curved forward process $x_t = \sqrt{\bar\alpha_t}x_0 + \sigma_t\epsilon$. You only need it for DanceGRPO's DDPM branch and the ELBO used by DGPO / the Diffusion-DPO precursor; those symbols are defined locally in those files (see the note in [NOTATION.md §2](../NOTATION.md)). Mentally, DDPM is "the same idea on a curvier path." Everything else here is flow matching.

---

**Next**: [grpo_basics.md](grpo_basics.md) — the RL machinery that gets bolted onto this model. Then [flow_grpo.md](../policy_gradient/flow_grpo.md), where the two meet.
