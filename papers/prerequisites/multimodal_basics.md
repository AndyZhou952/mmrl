# Prerequisite — Multimodal Generative Models (the thing we align)

> Notation: follows [NOTATION.md](../NOTATION.md). Scoped to the model facts diffusion RL assumes; framed around the VeRL-Omni training pipeline.

The papers here fine-tune **text-to-image / text-to-video** (and increasingly unified text+image) generators. This primer sketches what those models are so the algorithm pages can focus on the RL. The mental model: the generator is a **chef** — pretraining taught it to cook in general; RL is a **critic** that refines it toward a particular taste.

---

## 1. What we are aligning

A conditional generator maps a **prompt** $c$ (plus noise) to an image or video. Two parts matter here:

1. **Text encoder** — turns the prompt into the conditioning $c$ (a T5 / CLIP / VLM embedding). Treated as a black box.
2. **Generator** — a denoiser that, guided by $c$, walks noise to a sample. *This* is the part RL fine-tunes.

## 2. The VeRL-Omni pipeline: encoder → DiT → VAE

Modern systems do not paint pixels directly. A **VAE** compresses images/videos into a smaller **latent**; the generator works in latent space; the VAE decoder turns the final latent back into pixels. So throughout this repo $x_0$ is the **clean latent**, not raw pixels (video just adds a time axis to the latent).

In VeRL-Omni a single training **rollout** chains these heterogeneous parts — *text encoder → DiT → VAE decoder* — and, crucially, a rollout is a **"denoising trajectory in a continuous latent space"**, not a token sequence. That one reframing is why LLM-RL machinery has to be adapted rather than reused verbatim (and why scheduling is harder: each stage has a different memory profile).

## 3. The backbone: a conditional DiT

The generator is typically a **Diffusion Transformer (DiT / MMDiT)**: given the noisy latent $x_t$, the timestep $t$, and the conditioning $c$, it predicts a **velocity** $v_\theta(x_t,t,c)$ (flow matching) or noise $\epsilon_\theta$ (DDPM). How that prediction transports noise to data is the subject of [flow_matching_basics.md](flow_matching_basics.md). Production examples: SD3/SD3.5 and FLUX (image); HunyuanVideo, Wan, SkyReels (video); Qwen-Image / HunyuanImage 3.0 (unified). See [models.md](../../models.md) for who trains what with which algorithm.

## 4. Classifier-free guidance (CFG), and why RL dislikes it

At inference, quality is boosted by **classifier-free guidance**: run the network conditioned and unconditioned and extrapolate,

$$v_\text{CFG} = (1{+}w) v_\theta(x_t,t,c) - w v_\theta(x_t,t,\varnothing).$$

CFG matters for RL because it (a) costs a second forward pass per step and (b) is a blend of two networks that does not correspond to one clean per-step probability — so it muddies the importance ratio policy gradients rely on. That is why several methods ([DiffusionNFT](../direct_preference/diffusion_nft.md), [UniGRPO](../policy_gradient/uni_grpo.md)) deliberately train **CFG-free**.

## 5. Reward models: where the "taste" comes from

RL needs a scalar score $r(x_0, c)$. In VeRL-Omni these are **rule-based rewards** (e.g. an OCR exact-match) and **model-based rewards** (e.g. *VLM-as-judge*: a vision-language model reads the generated image and scores prompt adherence, aesthetics, or text accuracy). Treat the reward as a **black box that returns a number** — the algorithms differ in how they *use* that number, not how it is computed.

(The one exception worth knowing: [SRPO](../direct_preference/srpo.md) builds a *relative* reward from a frozen embedding model — the difference between a positive and a negative text condition — so it never needs to retrain the scorer as the policy drifts.)

## 6. Why RL at all — the gap this survey fills

Pretraining (flow matching on a huge corpus) gives broad capability but not **preference alignment**: prompt adherence, compositional correctness, legible text, smooth motion. RL fine-tuning uses the reward above to push the chef toward the critic's taste. *How* to run RL through a continuous, deterministic-ODE generator — a model with no natural per-step probabilities — is the entire subject of this repo.

---

**Next**: [flow_matching_basics.md](flow_matching_basics.md) — how the generator is trained and sampled, and the deterministic-ODE catch. Then [grpo_basics.md](grpo_basics.md) — the RL machinery.
