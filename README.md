# Multi-Modal Reinforcement Learning (MMRL)

A survey repository tracing the development of reinforcement learning algorithms for
multi-modal generative models — primarily text-to-image and text-to-video diffusion/flow models.

**Assumed background**: readers are expected to know (1) multimodal model architecture and training basics, (2) diffusion/flow matching training (DDPM, DDIM, rectified flow), and (3) RL fundamentals including PPO and GRPO as applied in LLMs. The papers covered here address how these RL ideas are extended to continuous generative models — a non-trivial gap that each paper solves differently.

---

## Repository Layout

```
mmrl/
├── README.md                     ← you are here
├── INTEGRATION_GUIDE.md          ← how to add new algorithms (format rules, notation, checklist)
├── models.md                     ← industry models using multimodal RL (2025–2026)
└── papers/
    ├── INDEX.md                  ← master table: all papers, dates, arXiv links, citation map
    ├── READING_GUIDE.md          ← how one approach leads to the next; recommended paths
    ├── advances.md               ← 20+ additional papers from late 2025–2026
    ├── coupled/                  ← SDE-coupled: training tied to stochastic sampling dynamics
    │   ├── flow_grpo.md          ← FlowGRPO (May 2025, NeurIPS 2025) — pioneering GRPO for flow matching
    │   ├── dance_grpo.md         ← DanceGRPO (May 2025) — concurrent; unified image+video
    │   ├── mix_grpo.md           ← MixGRPO + Flash variant (Jul 2025)
    │   ├── cps.md                ← CPS — noise-artifact fix for coupled methods (Sep 2025)
    │   ├── grpo_guard.md         ← GRPO-Guard — anti-reward-hacking for coupled (Oct 2025)
    │   └── uni_grpo.md           ← UniGRPO — reasoning-driven generation (Mar 2026)
    └── decoupled/                ← Solver-agnostic: training independent of sampling dynamics
        ├── srpo.md               ← SRPO (Sep 2025) — noise-prior recovery + semantic relative reward
        ├── diffusion_nft.md      ← DiffusionNFT — forward-process RL (Sep 2025)
        ├── awm.md                ← AWM — advantage-weighted matching loss (Sep 2025)
        └── dgpo.md               ← DGPO — group preference, ODE-compatible (Oct 2025)
```

Historical precursors (no dedicated files): **DDPO** (ICLR 2024, [2305.13301](https://arxiv.org/abs/2305.13301)) — first diffusion MDP, root of the coupled line; **Diffusion-DPO** (CVPR 2024, [2311.12908](https://arxiv.org/abs/2311.12908)) — offline DPO via diffusion ELBO, root of the decoupled line.

---

## Core Paradigm Split

The central conceptual divide in this field:

### Coupled paradigm

Training timesteps are **coupled with the SDE-based sampling dynamics**. Computing the policy gradient requires tractable log-probability $\log \pi_\theta(x_{t-1}|x_t)$ at each denoising step, which in turn requires the sampler to be stochastic (SDE-based) so that a density exists.

| What it means in practice | Consequence |
|---|---|
| Must use SDE samplers during training | Slower than deterministic ODE samplers |
| Importance ratio $\rho = \pi_\theta / \pi_{\theta_\text{old}}$ needed at each step | Sensitive to ratio imbalance across timesteps |
| Full reverse-process rollout required | Memory-intensive for long trajectories |

Methods: **FlowGRPO, DanceGRPO, MixGRPO** (and fixes: CPS, GRPO-Guard)

### Decoupled paradigm

Training timesteps are **decoupled from the actual sampling dynamics**. The training objective (preference loss, contrastive matching, or advantage-weighted matching) does not require log-probability computation over denoising steps, making these methods inherently **solver-agnostic** — any ODE solver can generate trajectories without modifying the training procedure.

| What it means in practice | Consequence |
|---|---|
| Can use fast ODE/DDIM samplers during training | Much faster data collection |
| No importance ratio needed | No ratio-imbalance failure mode |
| Works with black-box solvers and CFG | More compatible with production pipelines |

Methods: **Diffusion-DPO, DGPO, DiffusionNFT, AWM**

---

## Conceptual Taxonomy

### By training objective

| Objective type | Papers |
|---|---|
| **Policy gradient, PPO-clipped** | FlowGRPO, DanceGRPO, MixGRPO |
| **Group preference (DPO-style ELBO)** | Diffusion-DPO, DGPO |
| **Advantage-weighted matching loss** | AWM |
| **Contrastive forward-process** | DiffusionNFT |
| **Direct reward + semantic relative reward** | SRPO |

### By efficiency problem solved

| Problem | Solutions |
|---|---|
| Flow matching ODE has no density → GRPO undefined | FlowGRPO (ODE→SDE conversion) |
| All $T$ SDE steps need gradients → slow | FlowGRPO-Fast, MixGRPO, MixGRPO-Flash |
| SDE noise artifacts hurt reward learning | CPS |
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
2025-09  CPS            — DDIM-inspired sampler; eliminates SDE noise artifacts  [coupled fix]
2025-09  SRPO           — noise-prior closed-form recovery + semantic relative reward (Tencent)
2025-09  DiffusionNFT   — forward-process RL; no MDP, no SDE, no CFG conflicts
2025-09  AWM            — replaces DDPO log-prob with pretraining matching loss; 24× faster
2025-10  DGPO           — group preference via ELBO; ODE-compatible; ~20× faster
2025-10  GRPO-Guard     — ratio normalisation + gradient reweighting  [coupled fix]
2026-03  UniGRPO        — unified policy optimisation for reasoning-driven visual generation
```

See `papers/advances.md` for 20+ additional algorithm papers from late 2025–2026 (BranchGRPO, TreeGRPO, DenseGRPO, DiverseGRPO, DRIFT, TDM-R1, and more).

See `papers/INDEX.md` for the full citation graph.

---

## Recommended Reading Order

See **[`papers/READING_GUIDE.md`](papers/READING_GUIDE.md)** for the full narrative with explanations of how each paper leads to the next.

Quick paths:
- **Core chain** (how FlowGRPO is iteratively improved): `flow_grpo → mix_grpo → grpo_guard → cps`
- **Decoupled branch** (alternatives that drop the SDE requirement): `flow_grpo → awm → dgpo → srpo`
- **Concurrent breadth**: `dance_grpo` alongside either path above
