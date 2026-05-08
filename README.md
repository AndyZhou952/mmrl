# Multi-Modal Reinforcement Learning (MMRL)

A learning repository tracing the development of reinforcement learning algorithms for
multi-modal generative models — primarily text-to-image and text-to-video diffusion/flow models.

---

## Repository Layout

```
mmrl/
├── README.md                     ← you are here
├── models.md                     ← industry models using multimodal RL (2025–2026)
└── papers/
    ├── INDEX.md                  ← master table: all papers, dates, arXiv links, citation map
    ├── foundations/              ← 2023 precursors (before the coupled/decoupled split)
    │   ├── ddpo.md               ← DDPO (May 2023) — first diffusion MDP; root of coupled line
    │   └── diffusion_dpo.md      ← Diffusion-DPO (Nov 2023) — root of decoupled/DPO line
    ├── coupled/                  ← SDE-coupled: training tied to stochastic sampling dynamics
    │   ├── flow_grpo.md          ← Flow-GRPO + Fast variant (May 2025, NeurIPS 2025)
    │   ├── dance_grpo.md         ← DanceGRPO (May 2025)
    │   ├── mix_grpo.md           ← MixGRPO + Flash variant (Jul 2025)
    │   ├── cps.md                ← CPS — noise-artifact fix for coupled methods (Sep 2025)
    │   └── grpo_guard.md         ← GRPO-Guard — anti-reward-hacking for coupled (Oct 2025)
    └── decoupled/                ← Solver-agnostic: training independent of sampling dynamics
        ├── srpo.md               ← SRPO (Sep 2025) — noise-prior recovery + semantic relative reward
        ├── diffusion_nft.md      ← DiffusionNFT — forward-process RL (Sep 2025)
        ├── awm.md                ← AWM — advantage-weighted matching loss (Sep 2025)
        └── dgpo.md               ← DGPO — group preference, ODE-compatible (Oct 2025)
```

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

Methods: **DDPO, FlowGRPO, DanceGRPO, MixGRPO** (and fixes: CPS, GRPO-Guard)

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
| **Foundation MDP formulation** | DDPO |

### By efficiency problem solved

| Problem | Solutions |
|---|---|
| Diffusion has no discrete action space → PG undefined | DDPO (MDP framing), FlowGRPO (ODE→SDE), DanceGRPO |
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
2023-05  DDPO           — first diffusion MDP; policy gradient over denoising chain
2023-11  Diffusion-DPO  — offline DPO via diffusion ELBO (root of decoupled line)

         ─── ~18-month gap: LLM GRPO (DeepSeek-R1) proves out ───

2025-05  FlowGRPO       — GRPO for flow matching; ODE→SDE; +Fast variant  [NeurIPS 2025]
2025-05  DanceGRPO      — concurrent; unified image+video, 4 backbones
2025-07  MixGRPO        — sliding-window ODE/SDE; +Flash (−71% time)
2025-09  CPS            — DDIM-inspired sampler; eliminates SDE noise artifacts  [coupled fix]
2025-09  SRPO           — noise-prior closed-form recovery + semantic relative reward (Tencent)
2025-09  DiffusionNFT   — forward-process RL; no MDP, no SDE, no CFG conflicts
2025-09  AWM            — replaces DDPO log-prob with pretraining matching loss; 24× faster
2025-10  DGPO           — group preference via ELBO; ODE-compatible; ~20× faster
2025-10  GRPO-Guard     — ratio normalisation + gradient reweighting  [coupled fix]
```

See `papers/INDEX.md` for the full citation graph.

---

## Recommended Reading Order

1. `papers/foundations/ddpo.md` — the MDP framing all coupled methods rely on
2. `papers/foundations/diffusion_dpo.md` — the offline baseline for the decoupled line
3. `papers/coupled/flow_grpo.md` — the pivot point to GRPO-based coupled methods
4. `papers/coupled/dance_grpo.md` — concurrent, broader scope
5. Then choose your path:
   - **Coupled improvements**: MixGRPO → CPS → GRPO-Guard
   - **Decoupled methods**: AWM → DiffusionNFT → DGPO
