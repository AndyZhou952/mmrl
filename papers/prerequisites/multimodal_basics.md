# Prerequisite — Multimodal Generative Models

Before you can steer an image or video generator with reinforcement learning, it helps to know what you are steering. The methods in this survey fine-tune **text-to-image / text-to-video** (and increasingly unified text+image) generators; this primer sketches how those models are built so the algorithm pages can stay focused on the RL. The mental model to carry through: the generator is a **chef** — pretraining taught it to cook in general; RL is a **critic** that refines it toward a particular taste.

---

## What to expect

This page walks you through five things, in order:

1. **The architecture taxonomy** — the three families of multimodal networks the field actually trains, and what backbone each one uses.
2. **What RL fine-tunes** — within each architecture, the specific component (the generator/denoiser) that the algorithms in this survey touch.
3. **The latent-space mental model** — why $x_0$ in this repo means a *latent*, not pixels.
4. **Classifier-free guidance (CFG)** — what it is, why diffusion sampling uses it, and the real reason some RL methods drop it during training.
5. **The rollout stack** — who generates the training samples, and where VeRL-Omni actually sits in that picture.

---

## 1. Three families of multimodal generators

"Multimodal generative model" covers a wide range of architectures, and they are not interchangeable. It is cleanest to sort them by **how generation is wired into the network**. Three families dominate the models this survey aligns:

| Family | Backbone | Representative models | Typical tasks |
|---|---|---|---|
| **Omni-modality LM** | AR + DiT | Qwen3-Omni | any-to-any (text, image, audio, video in and out) |
| **Diffusion generator** | MM encoder + DiT + decoder | Qwen-Image, Wan2.2 | t2i, t2v, i2i, … |
| **Unified understanding & generation** | AR + specialized generator | BAGEL, HunyuanImage 3.0 | t2i, i2i, i2t, … |

The differences are real but the families also blur into each other, so it is worth seeing each one concretely.

**Omni-modality LM.** A single model perceives *and* produces across many modalities. [Qwen3-Omni](https://arxiv.org/abs/2509.17765) is the canonical example: a "Thinker-Talker" mixture-of-experts design where an autoregressive (AR) language core handles reasoning and text, and dedicated generation heads stream out other modalities. The "AR + DiT" label captures the family pattern — a token-predicting LM backbone paired with a separate generator for the continuous modalities (a diffusion/DiT-style head in the general case; Qwen3-Omni's own Talker streams audio through a lightweight causal codec) — so the same network can go any-to-any rather than serving one fixed task.

**Diffusion generator.** The workhorse for text-to-image and text-to-video. The pipeline is three staged parts: a **multimodal encoder** turns the prompt (and any reference image) into conditioning; a **Diffusion Transformer (DiT / MMDiT)** denoises in a compressed latent space; a **decoder** (a VAE decoder) turns the final latent back into pixels. [Qwen-Image](https://arxiv.org/abs/2508.02324) follows exactly this MLLM-encoder → MMDiT → VAE recipe, and [Wan2.2](https://github.com/Wan-Video/Wan2.2) does the same for video, adding a mixture-of-experts denoiser (a high-noise expert for early layout, a low-noise expert for late detail) on top of a high-compression video VAE. This is the family the policy-gradient and direct-preference algorithms were originally built around.

**Unified understanding & generation.** One model that both *reads* images (understanding, image-to-text) and *writes* them (generation, text-to-image). The backbone is an AR transformer paired with a specialized generator, often with separate pathways for the two jobs. [BAGEL](https://www.themoonlight.io/en/review/emerging-properties-in-unified-multimodal-pretraining) uses a Mixture-of-Transformer-Experts with one expert for understanding and one for generation sharing a single attention stream, plus separate visual encoders (a SigLIP-style encoder for understanding, a VAE for generation). [HunyuanImage 3.0](https://arxiv.org/abs/2509.23951) is a native multimodal MoE that models text tokens autoregressively while predicting image tokens through a diffusion-based head — a hybrid discrete-continuous design. These are the targets of the unified-model RL work (UniGRPO).

### What RL actually fine-tunes

Across all three families, the part the algorithms in this survey update is the **generator** — the denoiser that turns noise into a sample. The text/multimodal encoder is treated as a frozen black box that produces the conditioning $c$; the VAE/decoder is also fixed. RL touches the **velocity / noise predictor** inside the DiT (or the generation expert of a unified model). For omni and unified models that also emit text, the reasoning tokens can be optimized jointly with the image — that joint case is exactly what [UniGRPO](../policy_gradient/uni_grpo.md) addresses — but the image-side gradient still lands on the denoiser. So whenever a later page says "the policy," picture the DiT denoiser, not the whole stack.

## 2. The latent-space mental model

Modern systems do not paint pixels directly. A **VAE** compresses images/videos into a smaller **latent**; the generator works in latent space; the VAE decoder turns the final latent back into pixels. So throughout this repo $x_0$ is the **clean latent**, not raw pixels (video just adds a time axis to the latent).

The generator is typically a **Diffusion Transformer (DiT / MMDiT)**: given the noisy latent $x_t$, the timestep $t$, and the conditioning $c$, it predicts a **velocity** $v_\theta(x_t,t,c)$ (flow matching) or noise $\epsilon_\theta$ (DDPM). How that prediction transports noise to data is the subject of [flow_matching_basics.md](flow_matching_basics.md). The key reframing for RL: a sampling **rollout** here is a *denoising trajectory in a continuous latent space*, not a token sequence — which is why LLM-RL machinery has to be adapted rather than reused verbatim.

## 3. Classifier-free guidance (CFG)

A conditional diffusion model can, in principle, sample straight from $v_\theta(x_t,t,c)$. In practice that tends to under-use the prompt: samples drift generic and ignore details. **Classifier-free guidance** ([Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)) fixes this with a trick that costs almost nothing to train. During pretraining the condition is occasionally dropped (the prompt is replaced by an empty $\varnothing$), so the *same* network learns both a conditional and an unconditional predictor. At sampling time you run both and extrapolate away from the unconditional one:

$$v_\text{CFG} = (1{+}w) v_\theta(x_t,t,c) - w v_\theta(x_t,t,\varnothing).$$

The guidance scale $w$ trades diversity for prompt adherence: larger $w$ pushes harder toward the conditional signal, sharpening prompt-following and perceived quality. CFG is near-universal in production multimodal generation — it is the main reason text-to-image samples look "on prompt" — and it applies equally to the diffusion generators and the diffusion heads of unified models above.

**Why some RL methods drop CFG during training.** This is *not* a universal dislike, and the page should not claim it is. CFG and RL coexist fine in plenty of setups; the [direct-preference](../../README.md) methods that work on final samples are generally CFG-compatible. The friction is specific to **policy-gradient** methods that rely on a per-step **importance ratio** $\rho_t = \pi_\theta(x_{t-1}\mid x_t) / \pi_{\theta_\text{old}}(x_{t-1}\mid x_t)$. That ratio assumes each step is drawn from one well-defined per-step distribution. CFG breaks that assumption: the guided velocity $v_\text{CFG}$ is an extrapolated blend of two network outputs and does not correspond to the density of any single conditional policy, so the importance ratio computed from it is distorted or ill-defined. That is the reason [DiffusionNFT](../direct_preference/diffusion_nft.md) trains fully **CFG-free** — its stated problem is that "CFG modifies the effective policy non-trivially, invalidating the density computation" — and the reason [UniGRPO](../policy_gradient/uni_grpo.md) uses the single-pass conditional velocity in the training loop and applies CFG only at inference (which also saves the second forward pass per step). The takeaway: CFG is a sampling-time enhancement; the conflict is between CFG and the *importance ratio*, not between CFG and RL in general.

## 4. The rollout stack: who generates training samples

RL needs samples to score, and generating them is a separate concern from computing the loss. It is worth being precise about the pieces, because they are easy to conflate:

- **Rollout / inference engine** — the thing that actually runs the generation to produce a sample: a denoising trajectory in latent space for diffusion, or an LM decode for the autoregressive parts. Two engines can do this job. **diffusers** is the reference diffusion sampler. **vLLM-Omni** runs both the diffusion denoising rollout and LM decoding under one high-throughput serving stack (step-wise continuous batching, embedding caching, and similar tricks), reaching diffusers' accuracy at much higher throughput.
- **Reward engine** — scores each sample. These are **rule-based rewards** (e.g. an OCR exact-match) and **model-based rewards** (e.g. *VLM-as-judge*: a vision-language model reads the image and scores prompt adherence, aesthetics, or text accuracy). Treat the reward as a **black box that returns a number** — the algorithms differ in how they *use* the number, not how it is computed. (One exception worth knowing: [SRPO](../direct_preference/srpo.md) builds a *relative* reward from a frozen embedding model, so it never needs to retrain the scorer as the policy drifts.)
- **Training framework** — [**VeRL-Omni**](https://vllm.ai/blog/2026-05-14-verl-omni) is this layer, built on `verl` and `vllm-omni`. It is the **trainer that orchestrates** the loop, not the rollout engine itself: it currently drives **vLLM-Omni** for rollouts (chosen for high-throughput async serving at diffusers-level accuracy), invokes the reward engine to score, and runs the RL update. VeRL-Omni is also where this survey's two-family split (Policy Gradient vs Direct Preference) comes from, and it spans all three architecture families above — diffusion (Qwen-Image), AR+DiT omni (Qwen3-Omni), and unified (BAGEL, HunyuanImage 3.0).

The chained, heterogeneous rollout — *encoder → DiT → VAE decoder*, each stage with a different memory profile — is what makes scheduling harder than in text-only RL, and it is why the rollout produces a continuous trajectory rather than a token stream.

## 5. Why RL at all — the gap this survey fills

Pretraining (flow matching on a huge corpus) gives broad capability but not **preference alignment**: prompt adherence, compositional correctness, legible text, smooth motion. RL fine-tuning uses a reward to push the chef toward the critic's taste. *How* to run RL through a continuous, deterministic-ODE generator — a model with no natural per-step probabilities — is the entire subject of this repo. See [models.md](../../models.md) for who trains what with which algorithm.

---

**Next**: [flow_matching_basics.md](flow_matching_basics.md) — how the generator is trained and sampled, and the deterministic-ODE catch. Then [grpo_basics.md](grpo_basics.md) — the RL machinery.
