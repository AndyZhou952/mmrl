# Reading Guide — Tracing the Algorithm Development

A map of how the field developed: the order the algorithms arrived in, and what gap in the previous approach each one fills. It is deliberately brief — the depth lives on the individual pages. Read this to decide where to dive in.

---

## The Big Picture

Every algorithm here chases the same goal: **use a reward signal to make a generative model produce better images or videos**. The obstacle is that flow matching and diffusion models are not naturally policy-learnable — they do not define a stochastic policy with a tractable density, which is what RL algorithms need.

The field split into two families by how they resolve this, named after their **training objective** (following VeRL-Omni). The **Policy Gradient** family keeps a PPO-clip policy gradient over the denoising trajectory, which forces a stochastic (SDE) sampler. The **Direct Preference** family uses a preference or MSE loss on final samples, so it needs no SDE and any ODE sampler works. Within Direct Preference there is a finer split: **preference-based** methods (Diffusion-DPO, DGPO) rank samples against each other, while **reward-based** methods (SRPO, AWM) align directly to a reward or target without explicit pairs.

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
   ┌──────────┼───────────┬────────────┐
 MixGRPO  GRPO-Guard     CPS        FlowDPPO
 efficiency  stability  quality   trust region
```

---

## Step 0 — Prerequisites (skip if already fluent)

Three foundations run underneath everything that follows. If any one feels unfamiliar, start with its primer — each is scoped to exactly what this survey uses:

- [`prerequisites/multimodal_basics.md`](prerequisites/multimodal_basics.md) — what a text-to-image/video generator is (latent DiT, VAE, CFG, reward models).
- [`prerequisites/flow_matching_basics.md`](prerequisites/flow_matching_basics.md) — velocity training, the deterministic ODE sampler, and **why it has no per-step density** (the gap the whole field addresses).
- [`prerequisites/grpo_basics.md`](prerequisites/grpo_basics.md) — PPO clip, the GRPO group-relative advantage, and the LLM→diffusion vocabulary map.

---

## Part I — The Policy Gradient Chain

### FlowGRPO — the base

**Start here.** [`policy_gradient/flow_grpo.md`](policy_gradient/flow_grpo.md) — ODE→SDE conversion so GRPO gets a tractable per-step density.

### MixGRPO — efficiency

[`policy_gradient/mix_grpo.md`](policy_gradient/mix_grpo.md) — Sliding ODE/SDE window: only a few steps carry gradient, for speed.

### GRPO-Guard — stability

[`policy_gradient/grpo_guard.md`](policy_gradient/grpo_guard.md) — Normalizes the per-step ratio to revive PPO clipping and stop reward hacking.

### DanceGRPO — concurrent breadth

[`policy_gradient/dance_grpo.md`](policy_gradient/dance_grpo.md) — Concurrent unification spanning DDPM and flow matching across image and video backbones.

### CPS — sample quality

[`policy_gradient/cps.md`](policy_gradient/cps.md) — DDIM-style sampler kills SDE noise artifacts that mislead reward models.

### FlowDPPO — trust region without clipping

[`policy_gradient/flow_dppo.md`](policy_gradient/flow_dppo.md) — Exact closed-form KL trust-region mask instead of the PPO clip.

---

## Part II — The Direct Preference Branch

Every method in the policy-gradient chain accepts one premise: GRPO needs a stochastic policy, so the deterministic ODE must become an SDE. This branch refuses it — each method replaces the per-step importance ratio with a signal that needs no density, so any ODE/DDIM/DPM sampler can generate training samples. Once you understand *why* FlowGRPO needs the SDE, read this branch as a set of ways to avoid it.

### Diffusion-DPO — the root of the branch

**Read first for the lineage.** [`direct_preference/diffusion_dpo.md`](direct_preference/diffusion_dpo.md) — Offline DPO via the diffusion ELBO; root of the branch.

### AWM — the simplest break

[`direct_preference/awm.md`](direct_preference/awm.md) — Advantage-weighted clean-target matching, with no SDE and no importance ratio.

### DiffusionNFT — the forward-process route

[`direct_preference/diffusion_nft.md`](direct_preference/diffusion_nft.md) — Forward-process contrastive RL via positive/negative velocity fields; CFG-free.

### DGPO — online group preference

[`direct_preference/dgpo.md`](direct_preference/dgpo.md) — Online group preference via the ELBO; ODE-compatible.

### SRPO — practical efficiency

[`direct_preference/srpo.md`](direct_preference/srpo.md) — Noise-prior recovery plus relative reward, with no reward-model retraining.

---

## Part III — Beyond the Core

### UniGRPO — text + image joint optimisation (2026)

[`policy_gradient/uni_grpo.md`](policy_gradient/uni_grpo.md) — Joint text+image policy optimization for unified models, as a single MDP.

### FlowDPPO — exact-KL trust region instead of ratio clipping (2026)

**Read after** FlowGRPO and GRPO-Guard. [`policy_gradient/flow_dppo.md`](policy_gradient/flow_dppo.md) — Exact-KL trust-region mask instead of the PPO clip.

### academia.md — the 2025–2026 long tail

[`academia.md`](academia.md) — The 2025–2026 long tail: 20+ shorter entries in the same format.

---

## Recommended Reading Paths

**Path A — Core chain (4–5 papers, ~2 hours).** `flow_grpo → mix_grpo → grpo_guard → cps`, then optionally DanceGRPO for the multi-backbone perspective.

**Path B — Direct Preference branch (3 papers, ~1.5 hours).** `flow_grpo (motivation only) → awm → dgpo → srpo`. Read FlowGRPO to see *why* the SDE is needed, then the direct-preference papers as ways to avoid it.

**Path C — Full survey.**

```
flow_grpo
    │
    ├── mix_grpo → grpo_guard → cps → flow_dppo   (policy-gradient improvements)
    ├── dance_grpo                                 (concurrent breadth)
    │
    ├── awm → diffusion_nft → dgpo                (direct-preference branch)
    ├── srpo                                       (practical fine-tuning)
    │
    └── uni_grpo → academia.md                     (2026 and beyond)
```

---

## Quick Reference: What Each Paper Changes

| Paper | What it keeps from predecessor | What it changes |
|---|---|---|
| **Diffusion-DPO** | DPO from LLMs | ELBO likelihood proxy → pairwise denoising-MSE margin (root of Direct Preference) |
| **FlowGRPO** | GRPO from LLMs (verbatim) | ODE→SDE conversion; group advantage |
| **MixGRPO** | FlowGRPO SDE + objective | SDE/gradient confined to a sliding window |
| **GRPO-Guard** | FlowGRPO sampling (unchanged) | RatioNorm + gradient reweighting in objective |
| **DanceGRPO** | FlowGRPO SDE + GRPO clip | DDPM support; T2V/I2V; τ-subsampling |
| **CPS** | FlowGRPO objective (unchanged) | SDE step formula (preserves flow coefficients) |
| **FlowDPPO** | FlowGRPO SDE + group advantage | Replaces PPO ratio-clip with a Gaussian-KL trust-region mask |
| **AWM** | GRPO group advantage | Replaces IS ratio with clean-target MSE |
| **DiffusionNFT** | Flow matching MSE base | Implicit positive/negative policies; forward direction |
| **DGPO** | GRPO group generation | Replaces IS ratio with ELBO preference |
| **SRPO** | Reward evaluation on $x_0$ | Closed-form recovery; relative reward (no reference) |
| **UniGRPO** | FlowGRPO SDE + RatioNorm + window | Joint text+image MDP; velocity-space KL |
| **FlowDPPO** | FlowGRPO SDE + ratio + group advantage | Replaces PPO clip with an exact Gaussian-KL asymmetric divergence mask |
