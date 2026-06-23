# Paper Index

Master reference for all papers in this repository.
Sorted chronologically. Citation edges: → means "is cited by".

For a narrative explanation of how one algorithm leads to the next, see [READING_GUIDE.md](READING_GUIDE.md).

---

## Paradigm overview

| Paradigm | Defining property (objective) | Consequence | Papers |
|---|---|---|---|
| **Policy Gradient** | PPO-clip / importance-weighted policy gradient summed over multiple denoising timesteps of the same trajectory | Needs per-step log-prob ⇒ SDE sampler | FlowGRPO, DanceGRPO, MixGRPO + fixes (CPS, GRPO-Guard), UniGRPO, FlowDPPO |
| **Direct Preference** | Preference / MSE-style loss on final (or single) samples; no per-step importance ratio | Solver-agnostic — any ODE/DDIM/DPM sampler | Diffusion-DPO, SRPO, DiffusionNFT, AWM, DGPO |

This taxonomy follows the **VeRL-Omni** *ubiquitous language* (`verl_omni/trainer/diffusion/diffusion_algos.py`): `flow_grpo`, `dance_grpo`, `flow_dppo`, `grpo_guard` register as the policy-gradient family; `dpo`, `diffusion_nft` as the direct-preference family. (The older "coupled / decoupled" axis was an SDE-vs-ODE framing of the same split.)

Reference: [Flow-Factory algorithm taxonomy](https://github.com/X-GenGroup/Flow-Factory/blob/main/guidance/algorithms.md)

Historical precursors (no dedicated files in this repo): **DDPO** ([2305.13301](https://arxiv.org/abs/2305.13301), ICLR 2024) — first diffusion MDP; **Diffusion-DPO** ([2311.12908](https://arxiv.org/abs/2311.12908), CVPR 2024) — offline DPO via diffusion ELBO.

---

## Master Table

| Short name | Full title | arXiv | Date | Venue | Notes file |
|---|---|---|---|---|---|
| **FlowGRPO** | Flow-GRPO: Training Flow Matching Models via Online RL | [2505.05470](https://arxiv.org/abs/2505.05470) | 2025-05-08 | NeurIPS 2025 | [policy_gradient/flow_grpo.md](policy_gradient/flow_grpo.md) |
| **SRPO** | Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference | [2509.06942](https://arxiv.org/abs/2509.06942) | 2025-09-08 | — | [direct_preference/srpo.md](direct_preference/srpo.md) |
| **DanceGRPO** | DanceGRPO: Unleashing GRPO on Visual Generation | [2505.07818](https://arxiv.org/abs/2505.07818) | 2025-05-12 | — | [policy_gradient/dance_grpo.md](policy_gradient/dance_grpo.md) |
| **MixGRPO** | MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE | [2507.21802](https://arxiv.org/abs/2507.21802) | 2025-07-29 | — | [policy_gradient/mix_grpo.md](policy_gradient/mix_grpo.md) |
| **CPS** | Coefficients-Preserving Sampling for RL with Flow Matching | [2509.05952](https://arxiv.org/abs/2509.05952) | 2025-09-07 | — | [policy_gradient/cps.md](policy_gradient/cps.md) |
| **DiffusionNFT** | DiffusionNFT: Online Diffusion Reinforcement with Forward Process | [2509.16117](https://arxiv.org/abs/2509.16117) | 2025-09-25 | — | [direct_preference/diffusion_nft.md](direct_preference/diffusion_nft.md) |
| **AWM** | Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models | [2509.25050](https://arxiv.org/abs/2509.25050) | 2025-09-29 | — | [direct_preference/awm.md](direct_preference/awm.md) |
| **DGPO** | Reinforcing Diffusion Models by Direct Group Preference Optimization | [2510.08425](https://arxiv.org/abs/2510.08425) | 2025-10-09 | — | [direct_preference/dgpo.md](direct_preference/dgpo.md) |
| **GRPO-Guard** | GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping | [2510.22319](https://arxiv.org/abs/2510.22319) | 2025-10-25 | — | [policy_gradient/grpo_guard.md](policy_gradient/grpo_guard.md) |
| **UniGRPO** | UniGRPO: Unified Policy Optimization for Reasoning-Driven Visual Generation | [2603.23500](https://arxiv.org/abs/2603.23500) | 2026-03-25 | — | [policy_gradient/uni_grpo.md](policy_gradient/uni_grpo.md) |

---

## Citation Graph

Each edge `A → B` means "A is cited by B" (B builds on A).

```
DDPO (2023-05, policy-gradient precursor — no file in this repo)
  → FlowGRPO, DanceGRPO, MixGRPO, CPS, DiffusionNFT, AWM, DGPO, GRPO-Guard

GRPO/PPO (DeepSeek-R1, January 2025 — not in this repo)
  → FlowGRPO, DanceGRPO, MixGRPO, DGPO

Diffusion-DPO (2023-11, direct-preference precursor — no file in this repo)
  → DGPO (extends group-level DPO to online setting)
  → [offline DPO comparison baseline for policy-gradient methods]

FlowGRPO (2025-05) — first to apply GRPO to flow matching
  → MixGRPO, CPS, DiffusionNFT, AWM, GRPO-Guard, DGPO

DanceGRPO (2025-05)
  → CPS, DiffusionNFT, GRPO-Guard, DGPO

MixGRPO (2025-07)
  → [cited by later efficiency comparisons]

CPS (2025-09)
  → [plug-in fix for policy-gradient methods; no downstream dependants yet]

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
  POLICY GRADIENT                DIRECT PREFERENCE
  (SDE as consequence)            (any ODE solver)
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

See also: [academia.md](academia.md) for 20 additional algorithm papers from late 2025–2026 (BranchGRPO, TreeGRPO, DenseGRPO, DiverseGRPO, DRIFT, TDM-R1, and more).

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
