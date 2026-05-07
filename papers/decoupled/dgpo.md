# DGPO — Reinforcing Diffusion Models by Direct Group Preference Optimization

> Notation: follows [NOTATION.md](../NOTATION.md) §6 (ELBO shorthand). Uses DDPM convention: $\epsilon_\theta$, but extends to flow matching. Positive group $\mathcal{G}^+$, negative group $\mathcal{G}^-$.

| Field | Value |
|---|---|
| **arXiv** | [2510.08425](https://arxiv.org/abs/2510.08425) |
| **Submitted** | 2025-10-09 |
| **Venue** | — (preprint) |
| **Authors** | Yihong Luo, Tianyang Hu, Jing Tang |
| **GitHub** | https://github.com/Luo-Yihong/DGPO |
| **Paradigm** | **Decoupled** — training uses ELBO over generated images; any ODE sampler works |
| **Cites** | FlowGRPO (2505.05470), DanceGRPO (2505.07818), DDPO, DPO (Rafailov 2023), Diffusion-DPO |

---

## Motivation

FlowGRPO and DanceGRPO are **coupled**: they need a stochastic (SDE) policy to compute $\rho_t = \pi_\theta / \pi_{\theta_\text{old}}$ at each denoising step. This forces the use of slow SDE samplers, even though the most efficient modern samplers (DDIM, DPM-Solver++) are deterministic ODEs.

DGPO asks: **can we get group-level RL signal without a policy gradient at all?** The answer is yes, by extending Diffusion-DPO from binary pairs to group-level rankings, and adapting it to an **online** setting where groups are generated fresh each iteration. No SDE, no importance ratio, no policy gradient — just an ELBO-based preference loss.

---

## Setting

- **Model**: diffusion or flow matching, parameterised by $\theta$; reference policy $\pi_{\theta_\text{ref}}$.
- **Online**: generate a group of $N$ images per prompt each iteration using **any ODE sampler**.
- **Reward**: $r(x_0^{(i)}, c) \in [0,1]$ for each generated image.
- **Groups**: partition by advantage sign:
  - $\mathcal{G}^+ = \{i : \hat A^{(i)} > 0\}$ (above-mean images)
  - $\mathcal{G}^- = \{i : \hat A^{(i)} \leq 0\}$ (below-mean images)

---

## Sampling (inference)

Any deterministic ODE sampler:
$$x_{t-\Delta t} = x_t - v_\theta(x_t, t, c)\,\Delta t \quad \text{(flow)} \quad \text{or} \quad x_{t-1} = \text{DDIM}_\theta(x_t, t, c)$$

**No SDE required.** This is the key efficiency gain.

---

## Reward and advantage calculation

Same group-relative normalisation as GRPO:
$$\hat A^{(i)} = \frac{r^{(i)} - \overline r}{\text{std}(\{r^{(j)}\}) + \delta}$$

Positive/negative partition:
$$\mathcal{G}^+ = \{i : \hat A^{(i)} > 0\}, \quad \mathcal{G}^- = \{i : \hat A^{(i)} \leq 0\}$$

Sample weights (must satisfy $\sum_{i \in \mathcal{G}^+} w^+(i) = \sum_{j \in \mathcal{G}^-} w^-(j)$ to cancel the partition function):
$$w^+(i) \propto |\hat A^{(i)}|, \quad w^-(j) \propto |\hat A^{(j)}|, \quad \text{renormalised}$$

---

## Training objective

### Step 1 — Log-ratio via ELBO

The intractable marginal log-ratio decomposes via the ELBO (see [NOTATION.md §6](../NOTATION.md)):
$$\log \frac{\pi_\theta(x_0 \mid c)}{\pi_{\theta_\text{ref}}(x_0 \mid c)} \approx -\mathbb{E}_t\!\left[\underbrace{\|\epsilon_\theta(x_t,t,c) - \epsilon\|^2}_{\mathcal{L}_\theta(x_0)} - \underbrace{\|\epsilon_{\theta_\text{ref}}(x_t,t,c) - \epsilon\|^2}_{\mathcal{L}_{\theta_\text{ref}}(x_0)}\right]$$

where $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sigma_t\,\epsilon$ and the expectation is over $t$ and $\epsilon$.

### Step 2 — Group Bradley-Terry objective

Model group preference with a Bradley-Terry log-likelihood:
$$\max_\theta\; \mathbb{E}\!\left[\log \sigma\!\left(R_\theta(\mathcal{G}^+ \mid c) - R_\theta(\mathcal{G}^- \mid c)\right)\right]$$

where the group-level reward proxy is:
$$R_\theta(\mathcal{G} \mid c) = \sum_{i \in \mathcal{G}} w(i) \cdot \log \frac{\pi_\theta(x_0^{(i)} \mid c)}{\pi_{\theta_\text{ref}}(x_0^{(i)} \mid c)}$$

The partition function of the Bradley-Terry model cancels when $\sum_{\mathcal{G}^+} w^+(i) = \sum_{\mathcal{G}^-} w^-(j)$, giving a tractable objective.

### Step 3 — Substituting the ELBO

Substituting the ELBO log-ratio:

$$\boxed{
\mathcal{L}_\text{DGPO}(\theta) = -\mathbb{E}_{t,\epsilon}\!\left[\log \sigma\!\left(\sum_{i \in \mathcal{G}^+} w^+(i)\,\Delta\mathcal{L}_i - \sum_{j \in \mathcal{G}^-} w^-(j)\,\Delta\mathcal{L}_j\right)\right]
}$$

where:
$$\Delta\mathcal{L}_i \triangleq \|\epsilon_{\theta_\text{ref}}(x_t^{(i)},t,c) - \epsilon^{(i)}\|^2 - \|\epsilon_\theta(x_t^{(i)},t,c) - \epsilon^{(i)}\|^2$$

(positive when $\theta$ is better than $\theta_\text{ref}$ at denoising $x_0^{(i)}$).

**Interpretation**: the loss encourages $\Delta\mathcal{L}_i > 0$ for positive-group images (model improves on ref) and $\Delta\mathcal{L}_j < 0$ for negative-group images (model gets worse on ref), weighted by advantage magnitude.

---

## Why no SDE is needed

DGPO's $\Delta\mathcal{L}_i$ is computed by:
1. Taking the generated $x_0^{(i)}$ (from ODE sampling — any sampler)
2. Independently sampling $t \sim \text{Uniform}\{t_\text{min}, T\}$ and $\epsilon^{(i)} \sim \mathcal{N}(0,I)$
3. Computing $x_t^{(i)} = \sqrt{\bar\alpha_t} x_0^{(i)} + \sigma_t \epsilon^{(i)}$ via the **forward process**
4. Evaluating the denoising MSE

Step 3 is the **forward** noising direction — independent of any sampling trajectory. No denoising steps, no importance ratio, no SDE needed.

---

## Training algorithm

```
Input: pretrained θ (= θ_ref initially), ODE sampler, reward r, prompt dist p_c
Repeat:
  1. Sample prompts {c_j}
  2. For each c_j, generate N images via ODE (any fast sampler):
       x_0^(1),...,x_0^(N) ~ ODE_θ(c_j)
  3. Compute rewards R^(i) = r(x_0^(i), c_j)
  4. Compute group advantages Â^(i) = (R^(i) - mean) / std
  5. Partition: G+ = {i: Â^(i)>0}, G- = {i: Â^(i)≤0}
  6. Compute weights w+(i), w-(j) ∝ |Â|, normalised to sum-match
  7. For each batch of (t, ε):
       t ~ Uniform{t_min,...,T};  ε^(i) ~ N(0,I)
       x_t^(i) = sqrt(ᾱ_t)·x_0^(i) + σ_t·ε^(i)     ← forward noising
       ΔL_i = ||ε_θ_ref(x_t^(i),t,c) - ε^(i)||² - ||ε_θ(x_t^(i),t,c) - ε^(i)||²
  8. L = -log σ(Σ_{G+} w+(i)·ΔL_i - Σ_{G-} w-(j)·ΔL_j)
  9. θ ← θ - η ∇_θ L
  10. Soft update reference: θ_ref ← θ periodically (or keep fixed)
```

---

## Comparison to coupled GRPO and Diffusion-DPO

| Aspect | Diffusion-DPO | FlowGRPO (coupled) | DGPO (decoupled) |
|---|---|---|---|
| Data source | Offline preference pairs | Online SDE rollouts | Online ODE rollouts |
| Preference signal | Binary ($x_0^w \succ x_0^l$) | Continuous advantage $\hat A^{(i)}$ | Group advantage + ranking |
| Sampler during training | Offline (no sampling) | SDE (mandatory) | Any ODE |
| Log-likelihood | ELBO over pairs | Per-step IS ratio | ELBO over group |
| Training speed vs. FlowGRPO | — | $1\times$ (baseline) | ~20× faster |
| Online adaption | No | Yes | Yes |

---

## Limitations

- Requires a reference policy $\pi_{\theta_\text{ref}}$; as $\theta$ drifts far from $\theta_\text{ref}$, the ELBO approximation degrades.
- Group ranking requires $N$ forward passes per prompt (parallelisable but $O(N)$ memory).
- ELBO-based log-ratio inherits the same timestep-weighting approximation as Diffusion-DPO.
