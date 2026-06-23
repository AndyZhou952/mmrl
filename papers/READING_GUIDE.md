# Reading Guide — Tracing the Algorithm Development

This guide is for readers who want to understand not just what each algorithm does, but **why it came next** and **what gap in the previous approach it fills**. Read it as a narrative of a field solving one problem at a time.

---

## The Big Picture

All algorithms in this repository share the same top-level goal: **use a reward signal to make a generative model produce better images or videos**. The central difficulty is that generative models (flow matching, diffusion) are not naturally policy-learnable — they do not define a stochastic policy with a tractable density, which is what RL algorithms require.

The field split into two paradigms depending on how each work resolves this. We name them by **training objective** (following VeRL-Omni): the **Policy Gradient** family keeps a PPO-clip policy gradient over the trajectory (and therefore needs an SDE), while the **Direct Preference** family uses a preference/MSE loss on final samples (and therefore needs no SDE). Within Direct Preference there is a further split worth keeping in mind: **preference-based** methods (Diffusion-DPO, DGPO) rank samples against each other, whereas **reward-based** methods (SRPO, AWM) align directly to a reward/target without explicit pairs.

```
                      FlowGRPO (May 2025)
                      first to apply GRPO to flow matching
                            │
              ┌─────────────┼──────────────┐
              │             │              │
     POLICY GRADIENT    DanceGRPO      DIRECT PREFERENCE
   (fix FlowGRPO's    (concurrent;   (abandon SDE entirely)
    own problems)      broader)
              │
     ┌────────┴────────┐
  MixGRPO          GRPO-Guard
  (efficiency)     (stability)
              │
             CPS
          (sample quality)
```

---

## Step 0 — Prerequisites (skip if already fluent)

Nothing here is strictly assumed. If any of the three foundations is unfamiliar, read the matching primer first — each is scoped to exactly what this survey uses:

- [`prerequisites/multimodal_basics.md`](prerequisites/multimodal_basics.md) — what a text-to-image/video generator is (latent DiT, VAE, CFG, reward models).
- [`prerequisites/flow_matching_basics.md`](prerequisites/flow_matching_basics.md) — velocity training, the deterministic ODE sampler, and **why it has no per-step density** (the gap the whole field addresses).
- [`prerequisites/grpo_basics.md`](prerequisites/grpo_basics.md) — PPO clip, the GRPO group-relative advantage, and the LLM→diffusion vocabulary map.

---

## Part I — The Policy Gradient Chain

### Step 1: FlowGRPO — the base

**Read first.** [`policy_gradient/flow_grpo.md`](policy_gradient/flow_grpo.md)

FlowGRPO is the pivot point for the entire field. Before it, GRPO (from DeepSeek-R1, January 2025) had only been applied to text. Applying it to flow matching models (SD3, FLUX) requires solving one structural problem: flow matching uses a **deterministic ODE**, which has no density — so the importance ratio $\rho_t = \pi_\theta / \pi_{\theta_\text{old}}$ that GRPO depends on is undefined.

FlowGRPO's answer: **convert the ODE to a stochastically equivalent SDE** using the Fokker-Planck equation. The SDE adds controlled noise that (a) preserves the marginal distribution at every timestep and (b) yields a Gaussian per-step density, making $\rho_t$ tractable.

After reading this, you understand the complete baseline:
- How to make a flow model stochastic enough for GRPO
- The group-relative advantage $\hat{A}^{(i)}$
- The PPO-clip training objective
- The FlowGRPO-Fast variant (one SDE branch instead of T steps)

**What FlowGRPO does not solve:** (1) running SDE for all $T$ steps is expensive; (2) the importance ratio is systematically left-shifted, disabling PPO's clip; (3) the SDE noise pushes samples off the flow manifold.

---

### Step 2: MixGRPO — fixing the sampling trajectory

**Read second.** [`policy_gradient/mix_grpo.md`](policy_gradient/mix_grpo.md)

The first efficiency problem with FlowGRPO: the SDE runs through **all $T$ denoising steps**, requiring gradient storage for all of them ($O(T \cdot d)$ memory). Additionally, SDE is a first-order sampler — it cannot use high-quality ODE solvers (DPM-Solver++) for the steps outside the gradient window.

MixGRPO's insight: **only the steps that contribute to the importance ratio actually need to be stochastic**. ODE steps always have $\rho_t = 1$, contributing zero gradient. Therefore, confine the SDE and gradient tracking to a contiguous **sliding window** $\mathcal{W}(l)$ of $w$ steps, and run a high-quality ODE everywhere else.

The window slides through the trajectory over training, so all timesteps accumulate updates — just not all at once.

| | FlowGRPO | MixGRPO |
|---|---|---|
| SDE steps | All $T$ | Window $w$ (default 4 of 25) |
| Gradient steps | All $T$ | $w$ only |
| ODE solver outside | N/A | DPM-Solver++ (Flash variant) |
| Speedup | 1× | Up to 71% time reduction (Flash) |

**Relationship to FlowGRPO**: MixGRPO keeps the same SDE derivation, same importance ratio formula, same GRPO-clip objective — it only changes *which steps* are SDE. The FlowGRPO-Fast idea (single branch) can be seen as the limiting case $w=1$ with random placement.

---

### Step 3: GRPO-Guard — fixing the systematic bias

**Read third.** [`policy_gradient/grpo_guard.md`](policy_gradient/grpo_guard.md)

After solving efficiency, the next problem is **training stability**. Empirical analysis of FlowGRPO reveals two statistical pathologies in the importance ratio:

1. **Left-shift**: the mean of $\rho_t^{(i)}$ is systematically below 1. PPO's clip band $[1-\epsilon, 1+\epsilon]$ is designed for $\mathbb{E}[\rho_t] \approx 1$; when the mean is 0.7, most positive-advantage samples already sit below $1-\epsilon$ and the clip never activates — updates are unconstrained.

2. **~20× variance across timesteps**: late denoising steps (small $\sigma_t$) produce much larger log-ratios than early steps, causing the gradient to be dominated by fine-detail timesteps and under-optimise coarse structure.

GRPO-Guard's two fixes are composable and add only a few lines of code on top of any policy-gradient method:
- **RatioNorm**: standardise $\log\rho_t^{(i)}$ within the group at each $t$, restoring zero mean and unit variance → clip engages as designed.
- **Gradient reweighting**: multiply each timestep's loss by $\delta_t = 1/\Delta t$, equalising gradient magnitudes across timesteps.

**Relationship to FlowGRPO and MixGRPO**: GRPO-Guard is a **pure objective modifier** — it changes nothing about sampling. It can be applied on top of FlowGRPO, DanceGRPO, or MixGRPO (within the window). The RatioNorm formula is later adopted directly by UniGRPO.

---

### Step 4: DanceGRPO — concurrent generalisation

**Read alongside or after Steps 1–3.** [`policy_gradient/dance_grpo.md`](policy_gradient/dance_grpo.md)

DanceGRPO is concurrent with FlowGRPO (submitted 4 days later, independently). It arrives at the same ODE→SDE conversion, but from a different starting point: the authors wanted a **single unified GRPO implementation** that works across all backbone types (DDPM-style and flow matching) and all modalities (T2I, T2V, I2V).

The technical contribution over FlowGRPO is the derivation of the same SDE for DDPM, not just flow matching — producing a unified importance ratio formula that is identical in form for both families. This means one training loop can serve SD (DDPM), FLUX (flow), and HunyuanVideo (flow, video).

DanceGRPO also introduces timestep **subsampling** (random $\tau$-fraction of steps instead of a fixed reduced count), which is its efficiency reduction strategy compared to FlowGRPO's fixed $T_\text{train}$.

**Relationship to FlowGRPO**: same core problem and same solution; different framing. The SDE derivations are equivalent in spirit; DanceGRPO's DDPM branch is an additional generalisation. CPS and GRPO-Guard identify problems that apply equally to both papers.

---

### Step 5: CPS — fixing the sample quality

**Read last in the policy-gradient chain.** [`policy_gradient/cps.md`](policy_gradient/cps.md)

Even with MixGRPO's efficiency and GRPO-Guard's stability, there is a quieter problem: the SDE-generated $x_0$ samples used to compute rewards are **slightly off the flow manifold**. FlowGRPO's SDE adds noise $s_t\sqrt{\Delta t}\epsilon_t$ on top of the Euler step, producing a total noise level at $t-\Delta t$ that exceeds $(t-\Delta t)$ — the level the rectified flow schedule specifies. Over many steps, this accumulation makes the image slightly noisier than a clean image should be. Reward models trained on clean images give unreliable scores for these off-manifold samples.

CPS fixes this with a **coefficients-preserving step**: decompose $x_{t-\Delta t}$ into clean-image, noise-direction, and fresh-noise components, and set the coefficients so the total noise level is exactly $t-\Delta t$. This is the flow-matching analogue of stochastic DDIM.

**Relationship to all policy-gradient methods**: CPS is a **drop-in replacement** for the SDE sampling step. The GRPO objective, advantage computation, and update rule are completely unchanged. It can be combined with MixGRPO (apply CPS inside the window) and GRPO-Guard (apply Guard to the objective). CPS does not improve speed or ratio stability — it specifically fixes image quality degradation caused by off-manifold samples.

---

### Policy Gradient Chain Summary

The five papers above form a complete, self-contained development arc. Each one takes the previous result and adds exactly one targeted fix:

```
FlowGRPO
│  solves: ODE has no density → convert to SDE
│  leaves open: expensive (all T steps), ratio left-shifted, samples off-manifold
│
├── MixGRPO (efficiency)
│     solves: all T steps need grad → sliding window of w steps
│     leaves open: ratio left-shifted, samples off-manifold
│
├── GRPO-Guard (stability)
│     solves: ratio left-shifted + 20× variance → RatioNorm + gradient reweighting
│     leaves open: samples off-manifold
│
├── DanceGRPO (breadth — concurrent)
│     solves: only works for flow → unify DDPM + flow; add T2V, I2V
│     same open problems as FlowGRPO
│
└── CPS (sample quality)
      solves: SDE adds uncompensated noise → coefficient-preserving step
      all other open problems remain (SDE still required)
```

---

## Part II — The Direct Preference Branch

The policy-gradient chain improvements all start from the same premise: **GRPO requires a stochastic policy, so we must convert the ODE to an SDE**. The direct-preference branch asks: what if we do not accept that premise?

All direct-preference methods can use any ODE sampler (DDIM, DPM-Solver++) for generating training images. They replace the importance ratio with an alternative signal that does not require per-step densities.

**When to read this branch**: after reading FlowGRPO (Step 1 above), you can jump directly here. The direct-preference papers explicitly position themselves against the SDE requirement; knowing FlowGRPO is enough to understand the motivation.

---

### AWM — the simplest conceptual break

**Read first in the direct-preference branch.** [`direct_preference/awm.md`](direct_preference/awm.md)

AWM starts from a theoretical diagnosis: DDPO's policy gradient is mathematically equivalent to doing flow matching with a **noisy target** — the stochastic denoised sample $x_{t-1}$ rather than the clean image $x_0$. This noisy target inflates gradient variance and diverges from pretraining.

The fix is almost embarrassingly simple: use the **clean velocity target** $u_t = x_0 - \epsilon$ (the pretraining target) and weight each image's loss by its group-relative advantage $\hat{A}^{(i)}$:

$$\mathcal{L}_\text{AWM} = \mathbb{E}\left[\hat{A}^{(i)}\Vert{}v_\theta(x_t^{(i)}, t, c) - u_t^{(i)}\Vert^2\right]$$

This is the LLM pretraining-to-PPO analogy made exact for diffusion: in LLMs, PPO = pretraining loss × advantage; AWM establishes the same for flow models. No SDE, no importance ratio, no reference model lookup per gradient step. Reported speedup: **24× vs. FlowGRPO** on GenEval.

---

### DiffusionNFT — the forward-process approach

**Read second in the direct-preference branch.** [`direct_preference/diffusion_nft.md`](direct_preference/diffusion_nft.md)

DiffusionNFT takes a different structural departure: instead of replacing the training target (like AWM), it moves the entire RL signal into the **forward (noising) direction**. Images are still generated by an ODE sampler, but the gradient flows through the standard flow matching objective applied to the forward process — not through any denoising steps.

The mechanism uses two implicit velocity fields — a "positive" version that moves in the direction of the current update and a "negative" version that moves in the opposite direction — weighted by the reward. High-reward images train the positive field; low-reward images train the negative field. This design is CFG-compatible and makes no assumptions about the sampler.

**Relationship to AWM**: both avoid SDE and use the clean flow matching objective as a base. The difference is that AWM directly weights the loss by advantage, while DiffusionNFT constructs implicit positive/negative policies and trains them contrastively. AWM is simpler to implement and analyse; DiffusionNFT's contrastive structure may handle binary/sparse rewards more naturally.

---

### DGPO — ELBO-based group preference

**Read third in the direct-preference branch.** [`direct_preference/dgpo.md`](direct_preference/dgpo.md)

DGPO extends Diffusion-DPO from binary offline preference pairs to **online group-level ranking**. It keeps the GRPO group generation (generate $N$ images per prompt, compute group advantage) but replaces the importance ratio with the diffusion ELBO:

$$\log\frac{p_\theta(x_0|c)}{p_{\theta_\text{ref}}(x_0|c)} \approx \mathbb{E}_t\left[\Vert\epsilon_{\theta_\text{ref}} - \epsilon\Vert^2 - \Vert\epsilon_\theta - \epsilon\Vert^2\right]$$

Forward-noised versions of the generated images are constructed independently (no dependence on the sampler), and the ELBO difference is computed as a group Bradley-Terry preference loss. Any ODE sampler works for generation.

**Relationship to AWM**: AWM directly advantages-weights the flow matching loss; DGPO converts the advantage into a group preference and uses the ELBO as a log-likelihood proxy. DGPO retains a reference policy and has the same ELBO approximation weakness as Diffusion-DPO; AWM has neither.

---

### SRPO — practical efficiency via direct alignment

**Read last, or independently.** [`direct_preference/srpo.md`](direct_preference/srpo.md)

SRPO is the most practically efficient of the direct-preference methods. It introduces two independent ideas, each addressing a different bottleneck:

1. **Direct-Align**: The forward process $x_t = (1-t)x_0 + t\epsilon_\text{gt}$ with a *fixed* known noise $\epsilon_\text{gt}$ allows closed-form recovery of $\hat{x}_0$ in a single step — no rollout, no multi-step denoising in the gradient path. One network call per timestep, all noise levels covered.

2. **SRPO Reward**: Rather than training a reward model, compute a relative score between a positive text condition ("photo-realistic image") and a negative one ("AI-generated image") applied to the same generated image. This score is self-normalising — no reward model re-training as the policy shifts.

SRPO is used as stage 4 of the HunyuanImage 3.0 training pipeline (after MixGRPO and before ReDA), demonstrating that the two ideas are complementary to the policy-gradient chain rather than competing.

---

### Direct Preference Branch Summary

```
FlowGRPO
│  problem: SDE required for ρ_t
│
├── AWM
│     approach: clean-target advantage-weighted flow matching
│     key idea: pretraining loss × advantage = RL alignment
│     no SDE, no reference lookup; 24× speedup
│
├── DiffusionNFT
│     approach: implicit positive/negative policies in forward direction
│     key idea: contrastive flow matching; CFG-compatible
│     no SDE, no importance ratio
│
├── DGPO
│     approach: online group preference via ELBO
│     key idea: extend Diffusion-DPO from offline pairs to online groups
│     no SDE; requires reference policy; ~20× speedup
│
└── SRPO
      approach: closed-form single-step recovery + relative reward
      key idea: known noise prior → bypass multi-step rollout entirely
      no SDE, no reference model, no reward re-training; <10 min / 32 GPUs
```

---

## Part III — Beyond the Core

### UniGRPO — text + image joint optimisation (2026)

[`policy_gradient/uni_grpo.md`](policy_gradient/uni_grpo.md)

**Read after** the full policy-gradient chain and at least one direct-preference method. UniGRPO targets **unified multimodal models** — transformers that generate both text reasoning chains and images in a single forward pass. It takes the FlowGRPO SDE approach (with a sliding window, following MixGRPO), applies RatioNorm from GRPO-Guard, removes CFG from training, and adds a velocity-space MSE regulariser. The key novelty is treating text and image generation as a single MDP with a shared terminal reward, so a better reasoning chain and a better image are jointly optimised.

### academia.md — the 2025–2026 long tail

[`academia.md`](academia.md)

After reading the core papers, `academia.md` tracks 20+ papers from late 2025 and 2026. Each entry follows the same problem/idea format but is shorter (overview-level). The policy-gradient advances (BranchGRPO, TreeGRPO, DenseGRPO, Pro-GRPO, DRIFT) mostly address variants of the efficiency and credit-assignment problems; the direct-preference advances (DAV, SQDF, VMPO, TDM-R1) explore richer theoretical frameworks.

---

## Recommended Reading Paths

### Path A — Core chain only (4–5 papers, ~2 hours)

```
flow_grpo → mix_grpo → grpo_guard → cps
```

Then optionally add DanceGRPO for the multi-backbone perspective.

### Path B — Direct Preference branch (3 papers, ~1.5 hours)

```
flow_grpo (motivation only) → awm → dgpo → srpo
```

Read FlowGRPO to understand *why* the SDE is needed; then read the direct-preference papers as alternatives that avoid it.

### Path C — Full survey

```
flow_grpo
    │
    ├── mix_grpo → grpo_guard → cps      (policy-gradient improvements)
    ├── dance_grpo                        (concurrent breadth)
    │
    ├── awm → diffusion_nft → dgpo       (direct-preference branch)
    ├── srpo                             (practical fine-tuning)
    │
    └── uni_grpo → academia.md           (2026 and beyond)
```

---

## Quick Reference: What Each Paper Changes

| Paper | What it keeps from predecessor | What it changes |
|---|---|---|
| **FlowGRPO** | GRPO from LLMs (verbatim) | ODE→SDE conversion; group advantage |
| **MixGRPO** | FlowGRPO SDE + objective | SDE/gradient confined to sliding window |
| **GRPO-Guard** | FlowGRPO sampling (unchanged) | RatioNorm + gradient reweighting in objective |
| **DanceGRPO** | FlowGRPO SDE + GRPO clip | DDPM support; T2V/I2V; τ-subsampling |
| **CPS** | FlowGRPO objective (unchanged) | SDE step formula (preserves flow coefficients) |
| **AWM** | GRPO group advantage | Replaces IS ratio with clean-target MSE |
| **DiffusionNFT** | Flow matching MSE base | Implicit positive/negative policies; forward direction |
| **DGPO** | GRPO group generation | Replaces IS ratio with ELBO preference |
| **SRPO** | Reward evaluation on $x_0$ | Closed-form recovery; relative reward (no reference) |
| **UniGRPO** | FlowGRPO SDE + RatioNorm + window | Joint text+image MDP; velocity-space KL |
