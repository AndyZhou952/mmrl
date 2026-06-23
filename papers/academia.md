# Academia Advances in Multimodal RL — Late 2025 & 2026

> Scope: algorithmic papers proposing new training objectives or sampling strategies for RL fine-tuning of image/video diffusion and flow models. Robotics, language-only, and discrete-diffusion papers excluded. Sorted chronologically within each paradigm group. Math uses unified notation from [NOTATION.md](NOTATION.md).

> **Papers with dedicated pages** (not listed here): DDPO, Diffusion-DPO, SRPO, FlowGRPO, DanceGRPO, MixGRPO, CPS, GRPO-Guard, DGPO, DiffusionNFT, AWM, UniGRPO.

---

## Master Index (sorted by date)

| Date | arXiv | Paper | Paradigm | Key Problem |
|---|---|---|---|---|
| 2025-09 | [2509.06040](https://arxiv.org/abs/2509.06040) | BranchGRPO | Policy Gradient | Inefficient sequential rollouts; sparse credit assignment |
| 2025-10 | [2510.00502](https://arxiv.org/abs/2510.00502) | DAV | Direct Preference | Mode collapse under standard KL-RL; no EM framework |
| 2025-10 | [2510.02692](https://arxiv.org/abs/2510.02692) | P-GRAFT | Direct Preference | RAFT methods lack unified theory; terminal-only shaping suboptimal |
| 2025-10 | [2510.18072](https://arxiv.org/abs/2510.18072) | AC-Flow | Direct Preference | No actor-critic framework for continuous-time flow matching |
| 2025-11 | [2511.19356](https://arxiv.org/abs/2511.19356) | TaRoS | Policy Gradient | Goodhart's Law / reward saturation in video GRPO |
| 2025-12 | [2512.04332](https://arxiv.org/abs/2512.04332) | DDRL | Direct Preference | On-policy KL regularisation causes reward hacking at scale |
| 2025-12 | [2512.04559](https://arxiv.org/abs/2512.04559) | SQDF | Direct Preference | No principled Q-function for diffusion denoising chain |
| 2025-12 | [2512.08153](https://arxiv.org/abs/2512.08153) | TreeGRPO | Policy Gradient | Trajectory-level uniform advantages; high sampling cost |
| 2025-12 | [2512.15347](https://arxiv.org/abs/2512.15347) | Pro-GRPO | Policy Gradient | Large groups costly; reward-clustered samples give no signal |
| 2025-12 | [2512.17951](https://arxiv.org/abs/2512.17951) | SuperFlow | Policy Gradient | Fixed group size ignores per-prompt variance; coarse step advantages |
| 2025-12 | [2512.21514](https://arxiv.org/abs/2512.21514) | DiverseGRPO | Policy Gradient | GRPO collapses toward few high-reward modes |
| 2025-12 | [2512.24138](https://arxiv.org/abs/2512.24138) | GARDO | Direct Preference | KL regularisation impedes exploration; no uncertainty-aware gating |
| 2026-01 | [2601.00423](https://arxiv.org/abs/2601.00423) | E-GRPO | Policy Gradient | Low-entropy SDE steps waste compute; exploration budget uniform |
| 2026-01 | [2601.04153](https://arxiv.org/abs/2601.04153) | Diffusion-DRF | Direct Preference | Non-differentiable rewards block VLM-based feedback |
| 2026-01 | [2601.05729](https://arxiv.org/abs/2601.05729) | TAGRPO | Policy Gradient | Standard GRPO fails on I2V; needs relational trajectory alignment |
| 2026-01 | [2601.12401](https://arxiv.org/abs/2601.12401) | DRIFT | Policy Gradient | RL fine-tuning provably converges to Dirac delta (mode collapse) |
| 2026-01 | [2601.20218](https://arxiv.org/abs/2601.20218) | DenseGRPO | Policy Gradient | Terminal reward applied uniformly to all steps — sparse signal |
| 2026-02 | [2602.12229](https://arxiv.org/abs/2602.12229) | VMPO | Direct Preference | SMC/variance perspective on diffusion alignment unexplored |
| 2026-03 | [2603.07700](https://arxiv.org/abs/2603.07700) | TDM-R1 | Direct Preference | RL for few-step (2–4 step) diffusion with non-differentiable rewards |
| 2026-03 | [2603.21872](https://arxiv.org/abs/2603.21872) | SAGE-GRPO | Policy Gradient | SDE noise in video generation degrades rollout quality off-manifold |

---

## Policy Gradient Paradigm Advances

Papers whose objective is a PPO-clip / importance-weighted policy gradient over the denoising trajectory (the SDE-based GRPO family) — or, more broadly, value/Q-based and reward-maximisation variants that optimise the same expected-reward objective.

---

### BranchGRPO — `2509.06040` · Sep 2025

**GitHub**: https://github.com/Fredreic1849/BranchGRPO

**Problem**: Sequential rollouts in GRPO are slow; applying the same trajectory-level reward to all timesteps gives no per-step credit signal; low-value branches waste compute.

**Approach**: Structures denoising as a **branching tree** rooted at a shared prefix. Branches split at selected timesteps to produce multiple candidate continuations from one shared trunk, amortizing early-step compute. Rewards are propagated upward via path-probability weighting to produce per-depth advantages:

$$\hat A_t^{(i)} = \frac{\sum_{b: t \in b} p(b)R^{(b)}}{\sum_{b: t \in b} p(b)} - \overline R$$

Only nodes in high-value branches receive gradient. Achieves ~55% compute reduction and +16% HPDv2.1 alignment vs. FlowGRPO.

---

### TaRoS — Target-Robust Reward Signaling · `2511.19356` · Nov 2025

**Problem**: Under sustained GRPO training, reward models over-fit to shortcut patterns (Goodhart's Law), especially in video generation where rewards span multiple quality axes. Reward saturation within groups removes the advantage signal.

**Approach**: **Intra-group sparse filtering** across reward components: decomposes the scalar reward into $K$ aspect scores $\lbrace r_k^{(i)}\rbrace$ and retains per-aspect advantage only for the top-$m$ samples per group (sparsity mask $\mathcal{S}_k$):

$$\hat A_k^{(i)} = \mathbf{1}[i \in \mathcal{S}_k]\cdot \frac{r_k^{(i)} - \overline r_k}{\text{std}(\lbrace r_k^{(j)}\rbrace) + \delta}$$

Combined advantage: $\hat A^{(i)} = \sum_k w_k \hat A_k^{(i)}$. Prevents any single reward axis from saturating; targets the weakest aspects per group.

---

### TreeGRPO — `2512.08153` · Dec 2025

**Website**: https://treegrpo.github.io

**Problem**: GRPO reuses the same scalar advantage for every denoising step in a trajectory — no fine-grained credit. BranchGRPO reduces compute but still uses path-aggregated rewards rather than true step-level estimates.

**Approach**: Recasts the full denoising process as a **search tree**. A single root noise $x_T$ branches into multiple children at strategically chosen timesteps; rewards are back-propagated through the tree to assign node-specific advantages:

$$\hat A_t^\text{node} = R(\text{subtree}(t)) - V(x_t)$$

where $V(x_t)$ is estimated by averaging sibling rewards. Multi-child branching yields multiple policy updates per forward pass. Achieves 2.4× training speedup vs. GRPO.

---

### Pro-GRPO — Expand and Prune · `2512.15347` · Dec 2025

**Problem**: Large group sizes $N$ reduce advantage variance but scale linearly in compute. Many samples cluster near the group mean (reward clustering), providing zero signal while consuming resources.

**Approach**: **Optimal Variance Filtering (OVF)**: first *expand* to a large candidate group $N_\text{large}$, then *prune mid-generation* (not post-hoc) using latent features to identify and discard reward-clustered trajectories. The surviving group $\lbrace x_0^{(i)}\rbrace_{i \in \mathcal{S}}$ has much higher per-sample variance:

$$\mathcal{S} = \arg\max_{|\mathcal{S}|=N_\text{keep}} \text{Var}(\lbrace r^{(i)}\rbrace_{i \in \mathcal{S}})$$

OVF uses intermediate latent features (not final images) to predict reward clustering, enabling early abortion of low-signal branches.

---

### SuperFlow — `2512.17951` · Dec 2025

**Problem**: FlowGRPO uses the same group size $N$ per prompt regardless of per-prompt reward variance. Trajectory-level advantages are biased because they ignore continuous-time flow dynamics at the step level.

**Key contributions**:
1. **Dynamic-Group sampling**: group size $N_c$ adapts per prompt based on estimated reward standard deviation $\hat\sigma_c$:

$$N_c = \left\lceil N_0 \cdot \frac{\hat\sigma_c}{\overline\sigma}\right\rceil$$
2. **Step-level advantage**: derives per-step advantage estimates from the flow ODE dynamics (continuous-time stochastic calculus) rather than reusing the terminal advantage at all steps.

Achieves +38.4% GenEval improvement over SD3.5-M baseline.

---

### DiverseGRPO — `2512.21514` · Dec 2025

**Website**: https://henglin-liu.github.io/DiverseGRPO/

**Problem**: Standard GRPO collapses to a few high-reward visual modes; the generated distribution loses semantic and stylistic diversity even as proxy reward improves.

**Key contributions**:
1. **Cluster-based exploratory reward**: spectral clustering over per-prompt generated images; samples in under-represented clusters receive a diversity bonus:

$$r_\text{div}^{(i)} = \frac{1}{|\mathcal{C}(i)|} \cdot r_\text{quality}^{(i)}$$
where $\mathcal{C}(i)$ is the cluster containing image $i$. Smaller clusters → larger bonus.
2. **Structure-aware regularisation**: stronger KL penalty at early denoising steps (high-$t$) to preserve structural diversity; relaxed at late steps where fine-grained reward signal matters.

---

### E-GRPO — Entropy-Aware GRPO · `2601.00423` · Jan 2026

**GitHub**: https://github.com/shengjun-zhang/VisualGRPO

**Problem**: GRPO applies SDE noise uniformly to all denoising steps. Low-entropy steps (where the model is near-deterministic) add noise but produce nearly indistinguishable rollouts, diluting reward signal. High-entropy steps (where the model is most uncertain) are under-exploited.

**Approach**: Merge consecutive low-entropy steps into one effective high-entropy SDE step; apply ODE to the remainder. Entropy is measured via the velocity field divergence $\nabla \cdot v_\theta$:

$$H_t = -\mathbb{E}[\log \pi_\theta(x_{t-\Delta t} | x_t)] \approx \frac{d}{2}\log(2\pi e\sigma_t^2\Delta t)$$

Steps with $H_t < H_\text{thresh}$ are merged. The merged step's noise is scaled to preserve total stochasticity:

$$\sigma_\text{merged} = \sqrt{\sum_{k \in \text{merged}} \sigma_{t_k}^2\Delta t}$$

Focuses exploration budget where it has most discriminative effect on reward.

---

### TAGRPO — Trajectory-Aligned GRPO for I2V · `2601.05729` · Jan 2026

**Problem**: Standard GRPO for image-to-video (I2V) generation provides only trajectory-level scalar advantages; video quality depends on temporal alignment between frames, which scalar rewards poorly capture.

**Approach**: **Contrastive trajectory alignment** in latent space. Within each group, high-advantage video latent trajectories are attracted toward each other; low-advantage trajectories are repelled from high-advantage ones:

$$\mathcal{L}_\text{contrast}(\theta) = -\sum_{i \in \mathcal{G}^{+}}\sum_{j \in \mathcal{G}^-} \log \frac{\exp(d(\mathbf{z}^{(i)}, \mathbf{z}^{(j)}) / \tau)}{\sum_{k} \exp(d(\mathbf{z}^{(i)}, \mathbf{z}^{(k)}) / \tau)}$$

where $\mathbf{z}^{(i)} = \lbrace x_{t_k}^{(i)}\rbrace_{t_k \in T_\text{SDE}}$ is the latent trajectory and $d(\cdot,\cdot)$ is a trajectory similarity. Combined with GRPO loss and a memory bank for diversity.

---

### DRIFT — Beyond Dirac Delta · `2601.12401` · Jan 2026

**Problem**: RL fine-tuning of diffusion models is **provably** converging to a Dirac delta (mode collapse) under standard PPO/GRPO objectives — demonstrated theoretically and empirically. No existing method has a principled fix.

**Approach**: Three-pronged diversity incentivisation within the GRPO framework:
1. **Reward-concentrated sampling**: retain only trajectories with reward $r^{(i)} \in [\overline r - \alpha\sigma, \overline r + \alpha\sigma]$ for gradient computation (removes outlier-induced mode pull).
2. **Stochastic prompt variation**: augment the conditioning $c$ with random variation $\tilde c = c + \delta c$ during rollout to expand the conditioning manifold.
3. **Potential-based reward shaping**: add a diversity potential $\Phi(x_0^{(i)})$ to the reward that measures distance to previously-generated samples, preventing the policy from collapsing to already-visited modes:

$$\tilde r^{(i)} = r^{(i)} + \lambda_\Phi\Phi(x_0^{(i)}), \quad \Phi(x_0) = \min_{j \in \mathcal{M}} d(x_0, x_0^{(j)})$$

---

### DenseGRPO — `2601.20218` · Jan 2026

**Problem**: All GRPO variants assign the same terminal reward $R(x_0, c)$ to every denoising step — a sparse credit assignment problem. The model cannot learn which steps contributed most to the final quality.

**Key contributions**:
1. **ODE-based intermediate reward estimation**: at each step $t_k$, compute the Tweedie clean-image prediction $\hat x_0(x_{t_k})$ and evaluate the reward model on it as a proxy for step $t_k$'s contribution:

$$r_{t_k}^{(i)} = R\left(\hat x_0(x_{t_k}^{(i)}), c\right), \quad \hat x_0 = x_t - tv_\theta(x_t,t,c)$$
2. **Reward-aware stochasticity calibration**: set the SDE noise coefficient $\sigma_{t_k}$ proportional to the expected reward gradient magnitude at step $t_k$, focusing exploration at high-impact steps.

Step advantage:

$$\hat A_{t_k}^{(i)} = \frac{r_{t_k}^{(i)} - \overline r_{t_k}}{\text{std}(\lbrace r_{t_k}^{(j)}\rbrace) + \delta}$$

---

### SAGE-GRPO — `2603.21872` · Mar 2026

**GitHub**: https://github.com/Tencent-Hunyuan/SAGE-GRPO

**Problem**: For video generation, ODE-to-SDE conversion injects noise that pushes intermediate states off the video data manifold; reward models trained on clean videos cannot reliably score these artifact-laden samples, destabilising RL training.

**Approach**: Defines the pre-trained flow model as implicitly specifying a **video manifold** and enforces dual proximity constraints:
- **Micro-constraint** (per step): the SDE noise is clipped so $x_{t-\Delta t}^\text{SDE}$ stays within $\delta_\text{micro}$ of the ODE trajectory $x_{t-\Delta t}^\text{ODE}$:

$$\Vert x_{t-\Delta t}^\text{SDE} - x_{t-\Delta t}^\text{ODE}\Vert_2 \leq \delta_\text{micro}$$
- **Macro-constraint** (rollout level): terminal frames must pass a quality gate ($R(x_0) \geq \tau_\text{macro}$) before contributing to the gradient.

Combined with FlowGRPO-style GRPO objective over the SDE window.

---

## Direct Preference Paradigm Advances

Papers whose objective is a preference / ELBO / matching loss on final (or single-step) samples — solver-agnostic, no per-step importance ratio. A few entries here are actor-critic, soft-Q, or reward-maximisation variants that are *solver-agnostic but reward-based* rather than preference-based per se; they are grouped here pending dedicated review and flagged in their entries.

---

### DAV — Diffusion Alignment as Variational EM · `2510.00502` · Oct 2025

**GitHub**: https://github.com/Jaewoopudding/dav

**Problem**: Standard KL-regularised RL for diffusion models converges to a single mode of the reward-tilted distribution (mode-seeking reverse KL). Multi-modal reward landscapes require a mode-covering objective.

**Approach**: Frames diffusion alignment as **iterative EM**:
- **E-step** (test-time search): find a set of diverse high-reward samples from the variational posterior $q(x_0) \propto r(x_0) \cdot \pi_{\theta_\text{old}}(x_0)$ via MCMC or rejection sampling.
- **M-step** (amortisation): minimise **forward KL** (mode-covering) from $q$ into $\pi_\theta$:

$$\mathcal{L}_\text{M}(\theta) = D_\text{KL}(q \Vert \pi_\theta) = -\mathbb{E}_{x_0 \sim q}[\log \pi_\theta(x_0)] + \text{const}$$

The forward KL is approximated via the ELBO (same trick as Diffusion-DPO):

$$\mathcal{L}_\text{M}(\theta) \approx \mathbb{E}_{x_0 \sim q,t,\epsilon}\left[\Vert v_\theta(x_t,t,c) - u_t\Vert^2\right]$$

Works for both continuous (T2I) and discrete (DNA) domains without differentiable reward.

---

### P-GRAFT — Intermediate Distribution Shaping · `2510.02692` · Oct 2025

**Venue**: ICLR 2026

**Problem**: RAFT-based methods (rejection sampling fine-tuning) shape only the terminal distribution $\pi(x_0)$; shaping at intermediate noise levels is theoretically suboptimal; no unified theory connects RAFT variants.

**Approach**: Introduces the **GRAFT framework** (Generalised RAFT) showing that all RAFT variants implicitly minimise:

$$\mathcal{L}_\text{GRAFT}(\theta) = D_\text{KL}(\tilde\pi \Vert \pi_\theta), \quad \tilde\pi(x_0) \propto \exp(R(x_0)/\beta)\pi_\text{ref}(x_0)$$

**P-GRAFT** extends this by shaping intermediate distributions $\tilde\pi_t(x_t)$ at noise level $t$, not just at $t=0$:

$$\mathcal{L}_\text{P-GRAFT}(\theta) = \sum_t \lambda_tD_\text{KL}(\tilde\pi_t \Vert \pi_\theta^t)$$

The bias-variance tradeoff is analysed: intermediate shaping has higher bias but lower variance than terminal-only shaping. Extended to flow model error correction.

---

### AC-Flow — Actor-Critic for Flow Matching · `2510.18072` · Oct 2025

**Website**: https://www.jiajunfan.com/projects/ac-flow/

**Problem**: Flow matching models lack a principled actor-critic framework that provides intermediate (per-step) feedback; naive critic regression on cumulative rewards is unstable.

**Approach**:
- **Critic**: trains a value function $V_\phi(x_t, t, c)$ on intermediate states via reward shaping for stable convergence:

$$V_\phi(x_t,t,c) = \mathbb{E}_{x_0 | x_t}\left[R(x_0,c)\right]$$
- **Wasserstein diversity regularisation**: penalises the 2-Wasserstein distance between the generated distribution and a reference in reward-weighted feature space:

$$\mathcal{L}_\text{div}(\theta) = W_2(\pi_\theta,\pi_\text{ref})$$
- **Dual stability**: advantage clipping + critic warm-up to prevent Q-value collapse in early training.

Actor update uses the intermediate advantage $\hat A_t^{(i)} = R^{(i)} - V_\phi(x_t^{(i)}, t, c)$ weighted by the flow matching loss.

---

### DDRL — Data-Regularized Diffusion RL · `2512.04332` · Dec 2025

**GitHub**: https://github.com/nvidia-cosmos/cosmos-rl

**Problem**: On-policy KL regularisation against a fixed reference policy is the primary cause of reward hacking (quality degradation, over-stylisation, diversity loss) in DDPO/FlowGRPO-style training. This is confirmed at scale (>1M GPU-hours of video generation training).

**Approach**: Replace on-policy KL with **forward KL to an off-policy data distribution** $p_\text{data}$. The standard diffusion training loss is exactly the forward KL:

$$\mathcal{L}_\text{diff}(\theta) = \mathbb{E}_{x_0 \sim p_\text{data},t,\epsilon}\left[\Vert v_\theta(x_t,t,c) - u_t\Vert^2\right] = D_\text{KL}(p_\text{data} \Vert \pi_\theta) + \text{const}$$

Combined objective:

$$\mathcal{L}_\text{DDRL}(\theta) = -\mathbb{E}_{x_0 \sim \pi_\theta}[R(x_0,c)] + \alpha\mathcal{L}_\text{diff}(\theta)$$

The reward term is on-policy; the data-regularisation term is off-policy, providing an unbiased anchor that prevents reward hacking. Scales to image and video generation.

---

### SQDF — Soft Q-based Diffusion Finetuning · `2512.04559` · Dec 2025

**GitHub**: https://github.com/Shin-woocheol/SQDF

**Problem**: Diffusion RL methods lack a principled soft Q-function for credit assignment over the long denoising chain; temporal discounting and multi-step value estimation are absent.

**Key contributions**:
1. **Training-free soft Q-function**: estimate $Q_\text{soft}(x_t, a_t)$ from the current model via the Bellman equation, using **consistency models** to efficiently predict $x_0$ from $x_t$ for value bootstrapping.
2. **Discounted denoising**: introduces discount factor $\gamma < 1$ so later denoising steps (closer to $x_0$) receive higher weight:

$$Q_t = r_t + \gammaV_{t-1}, \quad V_t = \mathbb{E}_{x_{t-1}|x_t}[Q_{t-1}]$$
3. **Off-policy replay buffer**: stores (trajectory, reward) pairs across iterations; combined with on-policy samples to improve mode coverage.

Policy gradient: $\nabla_\theta \mathcal{L} = \mathbb{E}[\nabla_\theta \log \pi_\theta(x_{t-1}|x_t) \cdot Q_\text{soft}(x_t, x_{t-1})]$.

---

### GARDO — Gated Adaptive Regularization · `2512.24138` · Dec 2025

**GitHub**: https://github.com/tinnerhrhe/GARDO

**Problem**: KL regularisation prevents reward hacking but also impedes exploration — a fundamental tension. Applying the same penalty everywhere regardless of local policy uncertainty is suboptimal.

**Approach**:
1. **Uncertainty-gated KL**: measure per-sample uncertainty $u^{(i)} = \text{Var}_{t,\epsilon}[\hat A^{(i)}(x_t)]$; apply KL penalty only where $u^{(i)} > u_\text{thresh}$:

$$\mathcal{L}_\text{GARDO}(\theta) = \mathcal{L}_\text{GRPO}(\theta) - \beta\mathbb{E}\left[\mathbf{1}[u^{(i)} > u_\text{thresh}]D_\text{KL}(\pi_\theta(x_0)\Vert\pi_\text{ref}(x_0))\right]$$
2. **Diversity-aware advantage shaping**: adds a novelty bonus to $\hat A^{(i)}$ for samples that differ from recently-generated images, encouraging exploration of new modes.

Acts as a plug-in wrapper compatible with any base GRPO/ELBO algorithm.

---

### VMPO — Variance Minimisation Policy Optimisation · `2602.12229` · Feb 2026

**Problem**: The standard KL-based alignment objective has been thoroughly studied, but the **variance** of the importance weights (a key driver of instability in SMC-type methods) has not been used as a policy optimisation target.

**Approach**: Frames diffusion alignment as **Sequential Monte Carlo (SMC)**: the denoising model is a proposal $\pi_\theta$; the reward-tilted distribution $\tilde\pi \propto R \cdot \pi_\text{ref}$ is the target; importance weights $w^{(i)} = \tilde\pi(x_0^{(i)})/\pi_\theta(x_0^{(i)})$ measure how well $\pi_\theta$ covers $\tilde\pi$.

**VMPO objective**: minimise the variance of log importance weights:

$$\mathcal{L}_\text{VMPO}(\theta) = \text{Var}_{x_0 \sim \pi_\theta}\left[\log\frac{\tilde\pi(x_0)}{\pi_\theta(x_0)}\right]$$

**Key theorem**: under on-policy sampling, $\nabla_\theta \mathcal{L}_\text{VMPO} = \nabla_\theta D_\text{KL}(\tilde\pi \Vert \pi_\theta)$, establishing equivalence with KL alignment on-policy. Off-policy, VMPO provides a correction term that reduces variance without mode-seeking bias.

---

### TDM-R1 — Trajectory Distribution Matching · `2603.07700` · Mar 2026

**GitHub**: https://github.com/Luo-Yihong/TDM-R1

**Problem**: RL for few-step (2–4 step) diffusion models. Non-differentiable rewards (binary human preference, object counts, OCR correctness) cannot be directly backpropagated; no surrogate reward learning framework exists for fast ODE samplers.

**Approach**: Decouples into two stages:
1. **Surrogate reward learning** via DGPO-style group preference optimisation on deterministic ODE trajectories. For each prompt, generate a group of $N$ images $\lbrace x_0^{(i)}\rbrace$; the surrogate reward $\hat R_\psi$ is trained to predict group-relative rankings using the ELBO log-ratio as the policy model:

$$\mathcal{L}_\text{surr}(\psi) = -\mathbb{E}\left[\log \sigma\left(\hat R_\psi(x_0^{+}) - \hat R_\psi(x_0^-)\right)\right]$$
2. **Generator training** guided by the surrogate:

$$\mathcal{L}_\text{gen}(\theta) = -\mathbb{E}_{x_0 \sim \pi_\theta}\left[\hat R_\psi(x_0,c)\right] + \betaD_\text{KL}(\pi_\theta \Vert \pi_\text{ref})$$

The key insight: ODE trajectory intermediate states $\lbrace x_{t_k}\rbrace$ provide unbiased, artifact-free intermediate value estimates (no SDE needed). Achieves GenEval 61% → 92% with only 4 denoising steps, surpassing the 40-step base model (63%) and GPT-4o (84%).

---

## Cross-Cutting Notes

**The exploration–exploitation tension** is now the central open problem. DiverseGRPO, DRIFT, GARDO, and DenseGRPO all attack it from different angles (diversity shaping, potential-based rewards, uncertainty-gated KL, dense credit assignment). No clear winner yet.

**Dense credit assignment** (TaRoS, DenseGRPO, TreeGRPO) is an active area — the mismatch between where gradients flow (denoising steps) and where reward is observed (terminal image) remains unsolved at the fundamental level.

**Few-step / distilled models** (TDM-R1) open a new front: RL methods optimized for many-step samplers may not transfer to 2–4 step consistency/distilled models.

**Video generation RL** (TaRoS, TAGRPO, SAGE-GRPO) is becoming a distinct sub-field as temporal consistency and off-manifold noise artifacts introduce qualitatively new challenges vs. image generation.
