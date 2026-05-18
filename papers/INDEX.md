# Paper Index

Master reference for all papers in this repository.
Sorted chronologically. Citation edges: → means "is cited by".

For a narrative explanation of how one algorithm leads to the next, see [READING_GUIDE.md](READING_GUIDE.md).

---

## Paradigm overview

| Paradigm | Defining property | Papers |
|---|---|---|
| **Coupled** | Training tied to SDE sampling; log-prob at each step required | FlowGRPO, DanceGRPO, MixGRPO + fixes (CPS, GRPO-Guard), UniGRPO |
| **Decoupled** | Solver-agnostic; no log-prob over denoising steps | Diffusion-DPO, SRPO, DiffusionNFT, AWM, DGPO |
| **Offline / other** | No online rollouts | Diffusion-DPO (offline) |

Reference: [Flow-Factory algorithm taxonomy](https://github.com/X-GenGroup/Flow-Factory/blob/main/guidance/algorithms.md)

Historical precursors (no dedicated files in this repo): **DDPO** ([2305.13301](https://arxiv.org/abs/2305.13301), ICLR 2024) — first diffusion MDP; **Diffusion-DPO** ([2311.12908](https://arxiv.org/abs/2311.12908), CVPR 2024) — offline DPO via diffusion ELBO.

---

## Master Table

| Short name | Full title | arXiv | Date | Venue | Notes file |
|---|---|---|---|---|---|
| **FlowGRPO** | Flow-GRPO: Training Flow Matching Models via Online RL | [2505.05470](https://arxiv.org/abs/2505.05470) | 2025-05-08 | NeurIPS 2025 | [coupled/flow_grpo.md](coupled/flow_grpo.md) |
| **SRPO** | Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference | [2509.06942](https://arxiv.org/abs/2509.06942) | 2025-09-08 | — | [decoupled/srpo.md](decoupled/srpo.md) |
| **DanceGRPO** | DanceGRPO: Unleashing GRPO on Visual Generation | [2505.07818](https://arxiv.org/abs/2505.07818) | 2025-05-12 | — | [coupled/dance_grpo.md](coupled/dance_grpo.md) |
| **MixGRPO** | MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE | [2507.21802](https://arxiv.org/abs/2507.21802) | 2025-07-29 | — | [coupled/mix_grpo.md](coupled/mix_grpo.md) |
| **CPS** | Coefficients-Preserving Sampling for RL with Flow Matching | [2509.05952](https://arxiv.org/abs/2509.05952) | 2025-09-07 | — | [coupled/cps.md](coupled/cps.md) |
| **DiffusionNFT** | DiffusionNFT: Online Diffusion Reinforcement with Forward Process | [2509.16117](https://arxiv.org/abs/2509.16117) | 2025-09-25 | — | [decoupled/diffusion_nft.md](decoupled/diffusion_nft.md) |
| **AWM** | Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models | [2509.25050](https://arxiv.org/abs/2509.25050) | 2025-09-29 | — | [decoupled/awm.md](decoupled/awm.md) |
| **DGPO** | Reinforcing Diffusion Models by Direct Group Preference Optimization | [2510.08425](https://arxiv.org/abs/2510.08425) | 2025-10-09 | — | [decoupled/dgpo.md](decoupled/dgpo.md) |
| **GRPO-Guard** | GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping | [2510.22319](https://arxiv.org/abs/2510.22319) | 2025-10-25 | — | [coupled/grpo_guard.md](coupled/grpo_guard.md) |
| **UniGRPO** | UniGRPO: Unified Policy Optimization for Reasoning-Driven Visual Generation | [2603.23500](https://arxiv.org/abs/2603.23500) | 2026-03-25 | — | [coupled/uni_grpo.md](coupled/uni_grpo.md) |

---

## Citation Graph

Each edge `A → B` means "A is cited by B" (B builds on A).

```
DDPO (2023-05, coupled precursor — no file in this repo)
  → FlowGRPO, DanceGRPO, MixGRPO, CPS, DiffusionNFT, AWM, DGPO, GRPO-Guard

GRPO/PPO (DeepSeek-R1, January 2025 — not in this repo)
  → FlowGRPO, DanceGRPO, MixGRPO, DGPO

Diffusion-DPO (2023-11, decoupled precursor — no file in this repo)
  → DGPO (extends group-level DPO to online setting)
  → [offline DPO comparison baseline for coupled methods]

FlowGRPO (2025-05) — first to apply GRPO to flow matching
  → MixGRPO, CPS, DiffusionNFT, AWM, GRPO-Guard, DGPO

DanceGRPO (2025-05)
  → CPS, DiffusionNFT, GRPO-Guard, DGPO

MixGRPO (2025-07)
  → [cited by later efficiency comparisons]

CPS (2025-09)
  → [plug-in fix for coupled methods; no downstream dependants yet]

SRPO (2025-09, Tencent)
  → [cited by: HunyuanImage 3.0 pipeline (stage 4); no downstream dependants in this repo yet]
```

Paradigm lineage:

```
  DDPO (2023, precursor)          Diffusion-DPO (2023, precursor)
  (MDP over denoising)            (offline DPO via ELBO)
         │                                  │
         │   + GRPO from LLMs               │
         │     (Jan 2025)                   │
         ▼                                  ▼
  COUPLED PARADIGM                DECOUPLED PARADIGM
  (SDE required)                  (any ODE solver)
  ─────────────────               ─────────────────
  FlowGRPO ─┬─ Fast variant       DGPO
  DanceGRPO ─┤                     DiffusionNFT
  MixGRPO ───┤─ Flash variant      AWM
             │                     SRPO
     ┌───────┴────────┐
     CPS          GRPO-Guard
   (noise fix)  (ratio fix)
```

---

## Key Problem/Solution Map

See also: [advances.md](advances.md) for 20 additional algorithm papers from late 2025–2026 (BranchGRPO, TreeGRPO, DenseGRPO, DiverseGRPO, DRIFT, TDM-R1, and more).

---

## Key Problem/Solution Map

| Problem | Paper that identified it | Solution |
|---|---|---|
| Diffusion denoising is not a standard MDP | (pre-DDPO literature) | DDPO: reframe as finite-horizon MDP |
| DPO can work offline for diffusion via ELBO | Diffusion-DPO | Rewrite DPO likelihood with diffusion ELBO |
| GRPO needs stochastic policy; flow uses deterministic ODE | FlowGRPO | ODE→SDE conversion preserving marginals |
| All $T$ steps need gradients → slow | FlowGRPO | "Fast" variant: branch once, 1–2 grad steps |
| All-step SDE still expensive | MixGRPO | Sliding window: SDE inside, ODE outside |
| SDE noise → artifacts → misleads reward model | CPS | DDIM-style coefficients-preserving sampler |
| Ratio mean < 1 and varying variance across steps | GRPO-Guard | RatioNorm + gradient reweighting |
| SDE blocks fast ODE samplers | DGPO / AWM / DiffusionNFT | Decouple training from sampling dynamics |
| DDPO's noisy-target loss diverges from pretraining | AWM | Advantage-weighted clean-target matching |
| Reverse-process MDP: solver restrictions + CFG conflicts | DiffusionNFT | Forward-process contrastive RL |
| Group preference without stochastic policy | DGPO | ELBO-based group preference (online DPO) |
