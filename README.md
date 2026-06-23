# Diffusion-RL Survey

A survey repository tracing the development of reinforcement learning algorithms for
multi-modal generative models — primarily text-to-image and text-to-video diffusion/flow models.

**Background — nothing is strictly assumed.** This survey aims to be self-contained for anyone with general ML maturity. The three foundations it builds on each have a short primer in `papers/prerequisites/`:

1. **Multimodal generative models** — architecture and training basics ([multimodal_basics.md](papers/prerequisites/multimodal_basics.md))
2. **Flow matching / diffusion** — DDPM, DDIM, rectified flow ([flow_matching_basics.md](papers/prerequisites/flow_matching_basics.md))
3. **GRPO / PPO from LLMs** — the RL machinery being transplanted ([grpo_basics.md](papers/prerequisites/grpo_basics.md))

If those are already familiar, skip straight to the papers — they focus on the non-trivial gap of extending these RL ideas to continuous generative models, which each paper solves differently. If not, read the relevant primer first; each is scoped to exactly the parts this survey needs.

This repository follows the **VeRL-Omni** *ubiquitous language*: algorithms are grouped by training objective into **Policy Gradient** and **Direct Preference** families (rather than the older "coupled / decoupled" SDE-vs-ODE framing), so a reader can move from "understand the algorithm" here to "run it in VeRL-Omni" directly.

---

## Repository Layout

```
diffusion-rl-survey/
├── README.md                     ← you are here
├── INTEGRATION_GUIDE.md          ← how to add new algorithms (format rules, notation, checklist)
├── models.md                     ← industry models using multimodal RL (2025–2026)
└── papers/
    ├── INDEX.md                  ← master table: all papers, dates, arXiv links, citation map
    ├── READING_GUIDE.md          ← how one approach leads to the next; recommended paths
    ├── academia.md               ← 20+ additional papers from late 2025–2026
    ├── prerequisites/            ← primers: GRPO basics, flow matching basics
    ├── policy_gradient/          ← PPO-clip policy gradient over the trajectory (FlowGRPO family)
    │   ├── flow_grpo.md          ← FlowGRPO (May 2025, NeurIPS 2025) — pioneering GRPO for flow matching
    │   ├── dance_grpo.md         ← DanceGRPO (May 2025) — concurrent; unified image+video
    │   ├── mix_grpo.md           ← MixGRPO + Flash variant (Jul 2025)
    │   ├── cps.md                ← FlowGRPO-CPS — noise-artifact fix (Sep 2025)
    │   ├── grpo_guard.md         ← GRPO-Guard — anti-reward-hacking (Oct 2025)
    │   └── uni_grpo.md           ← UniGRPO — reasoning-driven generation (Mar 2026)
    └── direct_preference/        ← preference / MSE loss on final samples (solver-agnostic)
        ├── srpo.md               ← SRPO (Sep 2025) — noise-prior recovery + semantic relative reward
        ├── diffusion_nft.md      ← DiffusionNFT — forward-process RL (Sep 2025)
        ├── awm.md                ← AWM — advantage-weighted matching loss (Sep 2025)
        └── dgpo.md               ← DGPO — group preference, ODE-compatible (Oct 2025)
```

Historical precursors (no dedicated files): **DDPO** (ICLR 2024, [2305.13301](https://arxiv.org/abs/2305.13301)) — first diffusion MDP, root of the policy-gradient line; **Diffusion-DPO** (CVPR 2024, [2311.12908](https://arxiv.org/abs/2311.12908)) — offline DPO via diffusion ELBO, root of the direct-preference line.

---

## Core Paradigm Split

The central conceptual divide in this field is the **training objective**:

### Policy Gradient paradigm

The objective is a **PPO-clip / importance-weighted policy gradient accumulated over multiple denoising timesteps of the same trajectory** (the FlowGRPO family). Computing it requires a tractable per-step log-probability $\log \pi_\theta(x_{t-1}\mid x_t)$, which in turn requires a stochastic (SDE) sampler so that a density exists — an *implication* of the objective, not its definition.

| What it means in practice | Consequence |
|---|---|
| Importance ratio $\rho = \pi_\theta / \pi_{\theta_\text{old}}$ needed at each step | Forces an SDE sampler; sensitive to ratio imbalance across timesteps |
| Gradient flows through (a window of) reverse steps | Slower / more memory-intensive than ODE inference |

Methods: **FlowGRPO, DanceGRPO, MixGRPO, FlowDPPO** (and fixes: FlowGRPO-CPS, GRPO-Guard); UniGRPO extends it to joint text+image.

### Direct Preference paradigm

The objective is a **preference or MSE-style loss evaluated on final (or single) samples** — preference, contrastive matching, or advantage-weighted matching — with **no per-step importance ratio**. This makes the methods inherently **solver-agnostic**: any ODE/DDIM/DPM solver can generate trajectories without changing the loss.

| What it means in practice | Consequence |
|---|---|
| Can use fast ODE/DDIM samplers during training | Much faster data collection |
| No per-step importance ratio | No ratio-imbalance failure mode |
| Works with black-box solvers and CFG | More compatible with production pipelines |

Methods: **Diffusion-DPO, DGPO, DiffusionNFT, AWM, SRPO**

---

## Conceptual Taxonomy

### By training objective

| Objective type | Family | Papers |
|---|---|---|
| **Policy gradient, PPO-clipped** | Policy Gradient | FlowGRPO, DanceGRPO, MixGRPO, FlowDPPO |
| **Group preference (DPO-style ELBO)** | Direct Preference | Diffusion-DPO, DGPO |
| **Advantage-weighted matching loss** | Direct Preference | AWM |
| **Contrastive forward-process** | Direct Preference | DiffusionNFT |
| **Direct reward + semantic relative reward** | Direct Preference | SRPO |

### By efficiency problem solved

| Problem | Solutions |
|---|---|
| Flow matching ODE has no density → GRPO undefined | FlowGRPO (ODE→SDE conversion) |
| All $T$ SDE steps need gradients → slow | FlowGRPO-Fast, MixGRPO, MixGRPO-Flash |
| SDE noise artifacts hurt reward learning | FlowGRPO-CPS |
| Ratio imbalance → reward hacking | GRPO-Guard |
| SDE requirement blocks fast ODE samplers | DGPO, AWM, DiffusionNFT |
| Pretraining objective diverges from RL objective | AWM |
| Reverse-process CFG conflicts | DiffusionNFT |
| Offline alignment (no online rollouts) | Diffusion-DPO |

---

## Development Timeline

```
2023-05  DDPO           — first diffusion MDP; policy gradient over denoising chain  [ICLR 2024, precursor]
2023-11  Diffusion-DPO  — offline DPO via diffusion ELBO  [CVPR 2024, precursor]

         ─── DeepSeek releases GRPO (January 2025) ───

2025-05  FlowGRPO       — first to apply GRPO to flow matching; ODE→SDE; +Fast variant  [NeurIPS 2025]
2025-05  DanceGRPO      — concurrent; unified image+video, 4 backbones
2025-07  MixGRPO        — sliding-window ODE/SDE; +Flash (−71% time)
2025-09  CPS            — DDIM-inspired sampler; eliminates SDE noise artifacts  [policy-gradient fix]
2025-09  SRPO           — noise-prior closed-form recovery + semantic relative reward (Tencent)
2025-09  DiffusionNFT   — forward-process RL; no MDP, no SDE, no CFG conflicts
2025-09  AWM            — replaces DDPO log-prob with pretraining matching loss; 24× faster
2025-10  DGPO           — group preference via ELBO; ODE-compatible; ~20× faster
2025-10  GRPO-Guard     — ratio normalisation + gradient reweighting  [policy-gradient fix]
2026-03  UniGRPO        — unified policy optimisation for reasoning-driven visual generation
```

See `papers/academia.md` for 20+ additional algorithm papers from late 2025–2026 (BranchGRPO, TreeGRPO, DenseGRPO, DiverseGRPO, DRIFT, TDM-R1, and more).

See `papers/INDEX.md` for the full citation graph.

---

## Recommended Reading Order

See **[`papers/READING_GUIDE.md`](papers/READING_GUIDE.md)** for the full narrative with explanations of how each paper leads to the next.

Quick paths:
- **Core chain** (how FlowGRPO is iteratively improved): `flow_grpo → mix_grpo → grpo_guard → cps`
- **Direct Preference branch** (alternatives that drop the SDE requirement): `flow_grpo → awm → dgpo → srpo`
- **Concurrent breadth**: `dance_grpo` alongside either path above
