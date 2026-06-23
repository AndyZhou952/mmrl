# Industry Models Using Multimodal RL (2025–2026)

Tracks **industry-grade production models** whose official technical reports document the use
of multimodal reinforcement learning in their training pipelines. Academia-only baselines are excluded.

Links to algorithm notes in this repo use relative paths (e.g. `papers/policy_gradient/flow_grpo.md`).

---

## Scope and conventions

- **RL algorithm links**: `[policy gradient]` or `[direct preference]` labels match the paradigm taxonomy in `papers/INDEX.md`.
- **"In-house"**: algorithm described in the model's technical report but with no separate public arXiv paper.
- **"Not disclosed"**: model uses RL/RLHF post-training but the specific algorithm is not named in the technical report.
- Unless noted, GitHub links point to the **official release repository**.

---

## Quick-reference table

| Model | Org | Date | Task | RL methods used |
|---|---|---|---|---|
| [Step-Video-T2V](#step-video-t2v) | StepFun | 2025-02 | T2V | Video-DPO |
| [HunyuanVideo](#hunyuanvideo-and-hunyuanvideo-15) | Tencent | 2025-12 | T2V, I2V | offline DPO + online RLHF |
| [HunyuanVideo 1.5](#hunyuanvideo-and-hunyuanvideo-15) | Tencent | 2025-11 | T2V, I2V | offline DPO + online RLHF |
| [Qwen2.5-Omni](#qwen25-omni) | Alibaba | 2025-03 | VLM + speech | DPO (offline) + GRPO (online) |
| [Wan 2.1](#wan-21-and-wan-22) | Alibaba | 2025-03 | T2V | not disclosed |
| [Wan 2.2](#wan-21-and-wan-22) | Alibaba | 2025-07 | T2V, I2V | RLHF (not disclosed) |
| [Seedream 4.0](#seedream-40) | ByteDance | 2025-09 | T2I + editing | RLHF (not disclosed) |
| [HunyuanImage 3.0](#hunyuanimage-30) | Tencent | 2025-09 | T2I | DPO + MixGRPO + SRPO + ReDA† |
| [Qwen-Image](#qwen-image) | Alibaba | 2025-08 | T2I + editing | DPO + GRPO |
| [Kling-Omni](#kling-omni) | Kuaishou | 2025-12 | T2V, I2V | DPO |
| [InternVL3.5](#internvl35) | Shanghai AI Lab | 2025-08 | VLM | MPO (offline) + GSPO (online) |

† = in-house algorithm, no separate public paper; see [New algorithms](#new-rl-algorithms-not-yet-in-this-repo).

---

## Models

---

### Step-Video-T2V

| Field | Value |
|---|---|
| **Organization** | StepFun |
| **Release date** | 2025-02-14 |
| **arXiv** | [2502.10248](https://arxiv.org/abs/2502.10248) |
| **GitHub** | https://github.com/stepfun-ai/Step-Video-T2V |
| **Task** | Text-to-video (T2V) |
| **Size** | 30 B parameters |
| **Base architecture** | Flow matching + 3D DiT, bilingual (EN/ZH) |

#### RL in training pipeline

| Stage | Algorithm | Purpose |
|---|---|---|
| Pre-training | — | Flow matching on large video corpus |
| SFT | — | Quality fine-tuning |
| Post-training | **Video-DPO** | Reduce motion artifacts, improve visual quality |

**Video-DPO**: a direct application of DPO to video generation. Preference pairs are constructed as (higher-quality video, lower-quality video) for the same prompt. The DPO loss is computed over the full video denoising trajectory. This is an in-house adaptation; no separate paper exists beyond the technical report. Conceptually extends Diffusion-DPO ([2311.12908](https://arxiv.org/abs/2311.12908)) [direct preference] to video.

---

### HunyuanVideo and HunyuanVideo 1.5

| Field | HunyuanVideo | HunyuanVideo 1.5 |
|---|---|---|
| **Organization** | Tencent | Tencent |
| **Release date** | 2024-12-03 | 2025-11-25 |
| **arXiv** | [2412.03603](https://arxiv.org/abs/2412.03603) | [2511.18870](https://arxiv.org/abs/2511.18870) |
| **GitHub** | https://github.com/Tencent-Hunyuan/HunyuanVideo | https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 |
| **Task** | T2V, I2V | T2V, I2V |
| **Size** | 13 B | 8.3 B |
| **Base architecture** | Flow matching DiT | Flow matching DiT (lighter) |

#### RL in training pipeline (1.5 report; 1.0 report did not detail RL)

| Stage | Algorithm | Purpose |
|---|---|---|
| CT | — | Continued pre-training |
| SFT | — | Quality fine-tuning |
| RLHF stage 1 (offline) | **DPO** [direct preference] | Preference pairs annotated for motion quality, semantics, aesthetics |
| RLHF stage 2 (online) | **Online RL** (not disclosed) | Further aesthetic and semantic refinement |

Applied separately for T2V and I2V pipelines. The offline DPO stage establishes a strong starting policy; the online stage (algorithm not named in the report) refines from there.

**SAGE-GRPO** (arXiv 2603.21872): a separate Tencent research paper fine-tunes HunyuanVideo-1.5 with manifold-aware GRPO [policy gradient], demonstrating state-of-the-art video alignment. This is a research demonstration, not the official HunyuanVideo training pipeline.

---

### Qwen2.5-Omni

| Field | Value |
|---|---|
| **Organization** | Alibaba (Qwen Team) |
| **Release date** | 2025-03 |
| **arXiv** | [2503.20215](https://arxiv.org/abs/2503.20215) |
| **GitHub** | https://github.com/QwenLM/Qwen2.5-Omni |
| **Task** | Multimodal understanding (text, image, audio, video) → text + streaming speech |
| **Size** | 7 B |
| **Base architecture** | Transformer (Thinker–Talker dual-track) |

#### RL in training pipeline

| Stage | Algorithm | Purpose |
|---|---|---|
| SFT | — | Instruction following |
| Offline RL | **DPO** [direct preference] (~150K pairs) | General alignment with human preferences |
| Online RL | **GRPO** (text-oriented) | Reasoning capability, objective QA, math |

Note: The RL here targets **multimodal understanding** (visual reasoning, speech synthesis quality, multimodal QA) — not image or video generation. GRPO is used in its text-LLM formulation, not the flow-matching adaptation tracked in `papers/policy_gradient/`.

---

### Wan 2.1 and Wan 2.2

| Field | Wan 2.1 | Wan 2.2 |
|---|---|---|
| **Organization** | Alibaba (Tongyi Lab) | Alibaba (Tongyi Lab) |
| **Release date** | 2025-03-26 | 2025-07 |
| **arXiv** | [2503.20314](https://arxiv.org/abs/2503.20314) | Covered in 2503.20314 (v2) |
| **GitHub** | https://github.com/Wan-Video/Wan2.1 | https://github.com/Wan-Video/Wan2.2 |
| **Task** | T2V, I2V | T2V, I2V |
| **Size** | 1.3 B and 14 B | A14B (MoE: 2 × 14B = 27B total) |
| **Architecture** | Flow matching DiT | Flow matching MoE-DiT |

#### RL in training pipeline

Wan 2.1 and Wan 2.2 technical reports focus on pre-training architecture (novel VAE, data curation, scaling laws). Post-training RL details are **not disclosed**.

Wan 2.2 introduces a **MoE architecture** with two denoising experts: a high-noise expert (layout/composition) and a low-noise expert (fine detail). The post-training process mentions RLHF but no algorithm name.

**Wan-R1** (third-party, not official): applies GRPO [policy gradient] on top of Wan2.2-TI2V-5B as a research demonstration of RL fine-tuning for image-to-video. Not part of Wan's official training pipeline.

---

### Seedream 4.0

| Field | Value |
|---|---|
| **Organization** | ByteDance (Seed Team) |
| **Release date** | 2025-09-24 |
| **arXiv** | [2509.20427](https://arxiv.org/abs/2509.20427) |
| **GitHub** | Not publicly released |
| **Task** | T2I + image editing + multi-image composition |
| **Base architecture** | Flow matching DiT |

#### RL in training pipeline

| Stage | Algorithm | Purpose |
|---|---|---|
| CT | — | Continued pre-training |
| SFT | — | Quality and instruction following |
| RLHF | **Not disclosed** | Align with human aesthetic preferences across text-to-image and editing tasks |

The paper confirms a multi-stage post-training pipeline with RLHF but does not name the specific algorithm (PPO / DPO / GRPO). Multiple reward models covering different quality dimensions are mentioned.

---

### HunyuanImage 3.0

| Field | Value |
|---|---|
| **Organization** | Tencent (Hunyuan Team) |
| **Release date** | 2025-09-28 |
| **arXiv** | [2509.23951](https://arxiv.org/abs/2509.23951) |
| **GitHub** | https://github.com/Tencent-Hunyuan/HunyuanImage-3.0 |
| **Task** | T2I (native multimodal, autoregressive LLM + flow matching image decoder) |
| **Size** | 80 B total (MoE), 13 B activated per token |
| **LMArena rank** | #1 on Text-to-Image leaderboard (as of report date) |

#### RL in training pipeline

The most detailed public post-training pipeline of any T2I model to date. Five sequential stages:

| Stage | Algorithm | Purpose |
|---|---|---|
| 1. SFT | — | Diverse high-quality images, progressive quality increase |
| 2. DPO | **Diffusion-DPO** [direct preference] | Suppress structural defects; pairs: high-quality vs. distorted |
| 3. MixGRPO | **MixGRPO** [policy gradient] | Aesthetic optimization; hybrid ODE-SDE sliding-window GRPO |
| 4. SRPO | **SRPO** [direct preference] | Semantic relative alignment: push images toward positive text descriptions |
| 5. ReDA† | **ReDA (in-house)** | Reward distribution alignment: minimize divergence from high-reward distribution |

**Algorithm notes:**
- **MixGRPO**: the same algorithm as `papers/policy_gradient/mix_grpo.md` — Tencent developed MixGRPO (arXiv 2507.21802) and applied it here.
- **SRPO**: the Tencent algorithm described in arXiv [2509.06942](https://arxiv.org/abs/2509.06942) — see `papers/direct_preference/srpo.md`. Injects a fixed noise prior into any timestep, recovers the clean image in one closed-form step, and scores with a semantic relative reward (positive vs. negative text condition difference). Direct Preference; no SDE or multi-step denoising gradients required.
- **ReDA**: see [New algorithms](#new-rl-algorithms-not-yet-in-this-repo).

---

### Qwen-Image

| Field | Value |
|---|---|
| **Organization** | Alibaba (Qwen Team) |
| **Release date** | 2025-08-04 |
| **arXiv** | [2508.02324](https://arxiv.org/abs/2508.02324) |
| **GitHub** | https://github.com/QwenLM/Qwen-Image |
| **Task** | T2I + image editing |
| **Base architecture** | MMDiT with dual encoder: Qwen2.5-VL (semantic) + VAE (appearance) |

#### RL in training pipeline

| Stage | Algorithm | Purpose |
|---|---|---|
| SFT | — | Curriculum learning: simple → complex text rendering |
| Post-training | **DPO** [direct preference] + **GRPO** [policy gradient] | Alignment with human aesthetic and text-accuracy preferences |

DPO and GRPO are confirmed in community reviews of the full technical report. The abstract focuses on curriculum learning for text rendering; algorithm details are in the post-training section of the PDF. Specific reward models used are not detailed in available public summaries.

---

### Kling-Omni

| Field | Value |
|---|---|
| **Organization** | Kuaishou |
| **Release date** | 2025-12-18 |
| **arXiv** | [2512.16776](https://arxiv.org/abs/2512.16776) |
| **GitHub** | Not publicly released |
| **Task** | T2V + I2V + multimodal understanding |
| **Base architecture** | Flow matching DiT |

#### RL in training pipeline

| Stage | Algorithm | Purpose |
|---|---|---|
| Post-training | **DPO** [direct preference] | Motion dynamics and visual integrity alignment |

The report explicitly favors DPO over GRPO to avoid the computationally expensive trajectory sampling that GRPO requires in flow-based models (i.e., the ODE→SDE conversion). Preference pairs are constructed via diverse condition sampling + human evaluation, then DPO loss is computed over the video diffusion trajectory.

This is a direct example of a production model **choosing the direct-preference paradigm** over policy-gradient (GRPO) for practical efficiency reasons.

---

### InternVL3.5

| Field | Value |
|---|---|
| **Organization** | Shanghai AI Lab |
| **Release date** | 2025-08-26 |
| **arXiv** | [2508.18265](https://arxiv.org/abs/2508.18265) |
| **GitHub** | https://github.com/OpenGVLab/InternVL |
| **Task** | VLM (multimodal understanding: image, video, text) |
| **Size** | Multiple sizes up to 78 B |
| **Base architecture** | InternViT + InternLM2 |

#### RL in training pipeline

| Stage | Algorithm | Purpose |
|---|---|---|
| Pre-training | — | Multimodal joint pre-training |
| SFT | — | Instruction following |
| Offline RL (warm-up) | **MPO** (Mixed Preference Optimization) | Efficient offline RL to establish good rollout quality |
| Online RL (refinement) | **GSPO** (Group Sequence Policy Optimization) | Online RL that refines output distribution |

**Cascade RL framework**: the offline stage acts as a warm-up providing high-quality rollouts; the online stage then refines precisely. Result: +16% reasoning performance, 4.05× inference speedup vs. InternVL3.

Note: InternVL3.5 uses RL for **multimodal understanding and reasoning**, not image generation. GSPO is a text-RL algorithm (developed for LLMs, similar in spirit to GRPO but designed for sequence-level optimization).

---

## New RL algorithms not yet in this repo

These algorithms appear in industry model training pipelines but have no separate arXiv paper, or limited public documentation. They are listed here for awareness; when more information becomes available they should be promoted to `papers/`.

---

### ReDA (Reward Distribution Alignment)

| Field | Value |
|---|---|
| **Proposed by** | Tencent (HunyuanImage 3.0 team) |
| **First appearance** | HunyuanImage 3.0 tech report ([2509.23951](https://arxiv.org/abs/2509.23951)), Sep 2025 |
| **Separate paper** | None (in-house algorithm, not published independently as of 2026-05) |
| **Paradigm** | Direct Preference (operates on final images, not per-step likelihood) |

**Description from the tech report**: ReDA minimizes the divergence between the model's output distribution and a "high-reward distribution" defined by a curated set of diverse high-quality images. Rather than using pairwise preferences (as in DPO) or policy gradient (as in GRPO), it directly aligns the model's generative distribution toward a reference distribution of visually excellent images.

**Key difference from DPO**: DPO compares individual pairs (preferred vs. dispreferred). ReDA compares the entire output distribution against a target distribution of high-quality images, making it closer to distribution matching / imitation learning than contrastive preference learning.

**Status**: Used as the final refinement stage in HunyuanImage 3.0 after MixGRPO and SRPO. The model achieves #1 on LMArena T2I leaderboard. No ablation comparing ReDA to alternatives is publicly available.

---

### Video-DPO

| Field | Value |
|---|---|
| **Proposed by** | StepFun (Step-Video-T2V team) |
| **First appearance** | Step-Video-T2V tech report ([2502.10248](https://arxiv.org/abs/2502.10248)), Feb 2025 |
| **Separate paper** | None (described only in the technical report) |
| **Paradigm** | Direct Preference (direct extension of Diffusion-DPO to video) |

**Description**: applies the Diffusion-DPO [direct preference] objective to video generation by constructing preference pairs of (higher-quality, lower-quality) videos for the same prompt, then computing the ELBO-based DPO loss over the video DiT. Reduces motion artifacts and improves temporal consistency.

Conceptually the same as Diffusion-DPO ([2311.12908](https://arxiv.org/abs/2311.12908)) extended to video. No theoretical novelty beyond the application; the technical report treats it as an engineering adaptation.

---

### GSPO (Group Sequence Policy Optimization)

| Field | Value |
|---|---|
| **Proposed by** | Alibaba (Qwen2.5-VL / InternVL3.5 reference) |
| **First appearance** | Used in Qwen2.5-VL and InternVL3.5 post-training |
| **arXiv** | Described in Qwen2.5-VL tech report ([2502.13923](https://arxiv.org/abs/2502.13923)) |
| **Scope** | Text LLM / VLM reasoning — **not** image generation |
| **Paradigm** | Text RL (out of scope for image/video generation papers in this repo) |

GSPO was developed to address instability in GRPO during long RL training runs (which can cause irreversible model collapse). It replaces the per-token ratio with a per-sequence ratio, reducing variance. Used in InternVL3.5 and Qwen2.5-VL post-training for reasoning.

**Not directly related** to the policy_gradient/direct-preference image-generation RL taxonomy in this repo, but listed here because InternVL3.5 uses it for multimodal reasoning alignment.

---

## RL usage patterns across the industry

### Offline DPO is universal

Every model with a disclosed post-training pipeline includes at least one DPO stage (often the first RL step). DPO serves as the stable, low-risk foundation layer before more aggressive online RL.

### Online RL follows offline DPO

HunyuanImage 3.0 (MixGRPO after DPO), HunyuanVideo 1.5 (online RL after DPO), Qwen2.5-Omni (GRPO after DPO), and InternVL3.5 (GSPO after MPO) all follow the same two-stage pattern: offline warm-up → online refinement.

### Direct Preference preferred for video; policy-gradient for image

Kling-Omni explicitly chose DPO over GRPO for video generation to avoid expensive SDE trajectory sampling. Step-Video-T2V also uses direct-preference Video-DPO. In contrast, HunyuanImage 3.0 uses policy-gradient MixGRPO for image generation, where the per-step SDE cost is more manageable.

### Algorithm disclosure is sparse

Most models (Seedream 4.0, Wan 2.1/2.2) confirm RLHF is used but do not name the specific algorithm. This is the norm rather than the exception in industry technical reports.
