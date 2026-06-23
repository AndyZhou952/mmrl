# Integration Guide — Adding New Algorithms to This Repository

This document is the authoritative reference for adding any new algorithm to this repository, whether done by a human contributor or an automated agent. Follow every rule here precisely. When in doubt, look at an existing paper file as a concrete example.

---

## 1. Scope — What Belongs Here

Include a paper **only if** it satisfies all three criteria:

| Criterion | Rule |
|---|---|
| **Domain** | RL fine-tuning of image or video generative models (diffusion or flow matching). Target tasks: T2I, T2V, I2V, or multimodal (text + image joint). |
| **Method class** | Proposes a new training objective, sampling strategy, reward design, or stability fix for aligning these models with reward signals. |
| **Novelty** | Introduces a technically distinct mechanism — not just an engineering application of an existing method to a new backbone. |

**Exclude** the following even if they cite papers in this repo:
- Language-only RL (no generative image/video component)
- Robotics or embodied RL
- Discrete diffusion (token-based image generation)
- Pure SFT / distillation with no RL reward signal
- Benchmark or evaluation papers (no new training method)

**When uncertain**: check whether the paper is cited by an existing paper in this repo, or whether it cites a paper that is already here. If neither, it probably does not belong.

---

## 2. Determining Where to Place the File

### Paradigm classification (most important judgment call)

Every paper in the policy-gradient or direct-preference directory must be classified:

| Paradigm | Test question (objective shape) | Directory |
|---|---|---|
| **Policy Gradient** | Is the objective a PPO-clip / importance-weighted **policy gradient accumulated over multiple denoising timesteps of the same trajectory**? (This forces per-step log-probs $\log\pi_\theta(x_{t-\Delta t}\mid x_t)$ ⇒ an SDE sampler — a *consequence*, not the criterion.) | `papers/policy_gradient/` |
| **Direct Preference** | Is the objective a **preference or MSE-style loss on final / near-final samples**, with no per-step importance ratio? (Any ODE/DDIM/DPM sampler then works unchanged.) | `papers/direct_preference/` |

The criterion is the **training objective**, matching VeRL-Omni's register names in `verl_omni/trainer/diffusion/diffusion_algos.py` (`flow_grpo`, `dance_grpo`, `flow_dppo`, `grpo_guard` → policy gradient; `dpo`, `diffusion_nft` → direct preference). SDE-vs-ODE is downstream of this choice.

**Decision rule in detail**:
- If the paper derives an importance ratio $\rho_t = \pi_\theta / \pi_{\theta_\text{old}}$ at individual denoising steps → **Policy Gradient**
- If the paper uses only the terminal images $\lbrace{}x_0^{(i)}\rbrace$ and a forward-noising or ELBO-based loss → **Direct Preference**
- If the paper is a pure fix/modifier (e.g., changes the objective formula but does not change the paradigm) → same directory as the base method it modifies

### Dedicated file vs. `academia.md` entry

| Use a dedicated file in `policy_gradient/` or `direct_preference/` when: | Add an entry to `academia.md` when: |
|---|---|
| The paper introduces a fundamentally new mechanism (new objective form, new SDE derivation, new paradigm insight) | The paper refines, extends, or ablates an existing mechanism without introducing a new structural approach |
| The paper is cited by multiple other papers already in the repo | The paper is a specialisation (e.g., applies FlowGRPO to a new task without changing the algorithm) |
| The paper is a primary citation target in industry model pipelines | The paper is mainly an empirical study with incremental results |

If unsure, prefer `academia.md` first; a dedicated file can always be created later when the paper proves influential.

**Promotion rule** (academia entry → dedicated page): promote an `academia.md` entry to its own file once **either** (a) it is cited by **≥ 2 papers already in this repo**, **or** (b) it is **adopted in a named industry pipeline** tracked in `models.md` (Tencent, Kuaishou, ByteDance, Alibaba, etc.). Until one of these holds, keep it as an overview-level entry in `academia.md`.

---

## 3. File Naming

```
papers/policy_gradient/<short_name>.md        e.g., flow_grpo.md, mix_grpo.md
papers/direct_preference/<short_name>.md      e.g., awm.md, dgpo.md
```

Rules for `<short_name>`:
- All lowercase, underscores for spaces
- Match the short name used in `INDEX.md`
- Abbreviate consistently with how the authors name their method (e.g., `grpo_guard` not `guard`, `diffusion_nft` not `nft`)

---

## 4. The MD File Format — Section by Section

Every dedicated paper file must follow this exact section order. Do not add, reorder, or rename sections.

---

### 4.1 Title Line

```markdown
# ShortName — Full Paper Title
```

- `ShortName`: the abbreviation used in the INDEX.md table (bold, 1–3 words)
- `Full Paper Title`: verbatim from the paper's arXiv title
- Separator: ` — ` (space–em-dash–space)

---

### 4.2 Notation Header (blockquote)

```markdown
> Notation: follows [NOTATION.md](../NOTATION.md)[, §N]. [Local overrides listed here.]
```

Rules:
- Always cite `[NOTATION.md](../NOTATION.md)` as the base
- If you use specific sections heavily, cite them: `§3 (flow matching)`, `§6 (ELBO shorthand)`, etc.
- After the NOTATION.md reference, list **only symbols that deviate from or extend NOTATION.md** — do not re-list symbols that are defined there
- Example additions: a paper-specific stochasticity parameter (`$\sigma_t$ (CPS stochasticity, $\in [0, t{-}\Delta t]$)`), a local group notation (`positive group $\mathcal{G}^{+}$`)
- If the paper uses no deviation from NOTATION.md, write: `> Notation: follows [NOTATION.md](../NOTATION.md).`

---

### 4.3 Metadata Table

```markdown
| Field | Value |
|---|---|
| **arXiv** | [NNNN.NNNNN](https://arxiv.org/abs/NNNN.NNNNN) |
| **Submitted** | YYYY-MM-DD (revised YYYY-MM-DD if applicable) |
| **Venue** | Conference name and year, or `— (preprint)` |
| **Authors** | Firstname Lastname, Firstname Lastname, ... |
| **GitHub** | full URL, or `—` if none |
| **Paradigm** | **Policy Gradient** or **Direct Preference** — one-sentence description |
| **Cites** | short names of papers this one cites that are relevant here |
| **Cited by** | short names of papers in this repo that cite this one |
```

Rules:
- arXiv ID must be a clickable link
- Submitted date: use the first submission date from arXiv; add `(revised ...)` if a major revision changed the content
- Venue: use the official short name (e.g., `NeurIPS 2025`, `ICLR 2024`, `CVPR 2024`)
- Authors: full names in order; do not truncate with "et al."
- GitHub: the canonical code repository linked in the paper; search the arXiv abstract page and the paper body; write `—` if genuinely absent
- Paradigm: bold the paradigm word; the description after `—` should state the defining property in one clause (not a sentence about what the paper is — a sentence about what structural property classifies it)
- Cites: only papers that are algorithmically relevant (the ones this paper positions against or builds on). Include arXiv IDs in parentheses when ambiguous.
- Cited by: fill from knowledge of the repo; update this field when a later paper is added

**Paradigm field examples:**
```
| **Paradigm** | **Policy Gradient** — per-step Gaussian log-prob required; must use SDE sampler |
| **Paradigm** | **Policy Gradient** — SDE + gradient confined to a sliding window; ODE everywhere else |
| **Paradigm** | **Direct Preference** — flow matching MSE reweighted by advantage; no SDE, no importance ratio |
| **Paradigm** | **Direct Preference** — ELBO-based group preference loss over online ODE rollouts |
```

---

### 4.4 Context or Background Section

Every file has exactly one second section. Choose between:

**`## Context`** — for papers that build directly on an existing paper in the repo:
- One paragraph maximum
- State what the predecessor papers established
- State what specific gap this paper addresses
- Link to the prerequisite files with relative paths: `[flow_grpo.md](flow_grpo.md)`
- Do not re-derive prerequisites — link to them

**`## Background: [Subtitle]`** — only for pivot papers that introduce two independent lines of prior work that readers may not connect:
- One subsection per ingredient (e.g., `### Flow matching`, `### GRPO for LLMs`)
- End with a subsection `### Why combining them is non-trivial`
- Readers are assumed to know: multimodal model basics, DDPM/flow matching training, PPO and GRPO as applied in LLMs

Do not use `## Motivation` — this was the old format and has been replaced by these two forms.

---

### 4.5 Problem Sections

These are the core of every paper file. Structure:

```markdown
## Problem N — [Short descriptive title]

**Issue**: [One or two sentences. What breaks in prior work, or what cannot be done. Quantify where possible.]

**Idea**: [What the paper proposes. State the mechanism concisely, not the motivation.]

**Why this works**: [The mathematical or structural reason. This is where you cite the theorem, duality, or property that makes the idea valid.]

**Result**: [The empirical evidence this idea produces. Quantitative (benchmark delta, speedup, ablation) or qualitative/descriptive, with the paper's table/figure number where available — e.g. "+12.3 GenEval over SD3.5-M (Tab. 2)" or "removes the colour-shift artifact visible in Fig. 4".]

[Mathematical derivation follows if needed — see §5.]
```

Rules:
- Number problems starting from 1: `## Problem 1 —`, `## Problem 2 —`, etc.
- The title after `—` is 4–8 words describing the specific bottleneck, not the solution
- **Issue** identifies the specific failure mode, not the general goal
- **Idea** describes the mechanism, not the motivation ("convert ODE to SDE", not "to make GRPO applicable")
- **Why this works** must state a reason — never leave it as a restatement of the idea
- **Result** grounds the idea in evidence from the paper: cite a number or a named figure/table; if the paper reports no isolated result for this specific idea (e.g. it is only validated in aggregate), say so and point to the consolidated `## Results` section. Do not invent numbers — pull them from the paper's arXiv/HTML, project page, or repo.
- If a problem has two sub-ideas (e.g., base variant + fast variant), use `**Idea 1 —**` and `**Idea 2 —**` sub-labels
- One section per independently-addressed problem; a paper with three distinct fixes has three Problem sections
- After the three bold blocks, include the mathematical derivation for the idea

---

### 4.6 Math Rules

All mathematical content must be consistent with `NOTATION.md`. Before writing any formula:

1. **Check NOTATION.md first.** If a symbol is defined there, use it exactly as defined, even if the paper uses a different symbol.
2. **If the paper uses a different symbol**, map it to the NOTATION.md symbol in the notation header (§4.2) — do not introduce the paper's symbol into the body.
3. **If the paper introduces a genuinely new symbol** (not covered in NOTATION.md), define it in the notation header. If the symbol will be used across multiple papers, add it to NOTATION.md under the appropriate section.

**Core symbol mappings** (these are fixed and must never be changed in paper bodies):

| Concept | Flow matching | DDPM |
|---|---|---|
| Model output | $v_\theta(x_t, t, c)$ | $\epsilon_\theta(x_t, t, c)$ |
| Noisy state | $x_t = (1-t)x_0 + t\epsilon$ | $x_t = \sqrt{\bar\alpha_t}x_0 + \sigma_t\epsilon$ |
| Clean image | $x_0$ | $x_0$ |
| Clean-image estimate | $\hat{x}_0 = x_t - tv_\theta$ | $\hat{x}_0 = (x_t - \sigma_t\epsilon_\theta)/\sqrt{\bar\alpha_t}$ |
| Score function | $\nabla_{x_t}\log p_t \approx (\hat{x}_0 - x_t)/t^2$ | $\nabla_{x_t}\log p_t \approx -\epsilon_\theta/\sigma_t$ |
| Importance ratio | $\rho_t^{(i)} = \pi_\theta / \pi_{\theta_\text{old}}$ | same |
| Group advantage | $\hat{A}^{(i)}$ (see NOTATION.md §5) | same |
| Per-step density (flow SDE) | $\pi_\theta(x_{t-\Delta t}\Vert{}x_t, c) = \mathcal{N}(\mu_\theta, \sigma_t^2\Delta tI)$ | $\pi_\theta(x_{t-1}\Vert{}x_t,c) = \mathcal{N}(\mu_\theta, \tilde\beta_t I)$ |
| KL penalty | $\betaD_\text{KL}(\pi_\theta \Vert \pi_\text{ref})$ | same |

**Display math rules**:
- Inline: `$...$` for single symbols and short expressions
- Display: `$$...$$` for equations that stand alone
- The main training loss of the paper **must** appear in a `$$\boxed{...}$$` block
- All other key equations use plain `$$...$$`
- Use `\mathrm` for operator names: `\mathrm{std}`, `\mathrm{clip}`, `\mathrm{KL}`
- Use `\Vert` for norms (not `\|` or `||`)
- Subscript `\text{old}`, `\text{ref}`, `\text{train}`, `\text{inf}` for parameter roles

---

### 4.7 Training Objective Section

```markdown
## Training Objective

[One sentence connecting this objective to the problems it solves.]

$$\boxed{
\mathcal{L}_\text{MethodName}(\theta) = ...
}$$

where [define each new symbol that appears in the boxed formula and was not defined in the Problem sections above].
```

Rules:
- Section heading is always `## Training Objective` (not "Loss", "Objective Function", etc.)
- The boxed formula is the paper's primary training loss — the one you would implement first
- If the paper has two separate objectives (e.g., UniGRPO text + image), use sub-subsections under this heading and box each one
- Always explain every symbol that appears in the box that hasn't been introduced yet
- The sentence before the box should say what problems (by number) the objective combines

---

### 4.8 Algorithm Section

```markdown
## Algorithm

```
Input: [parameters the caller configures before training starts]
[Initialize: any state set once before the loop]
Repeat:
  1. [Step]
  2. [Step]
     Sub-step details indented
  [variant or extension in a clearly labelled block at the end]
```
```

Rules:
- Section heading is always `## Algorithm`
- The block is a plain fenced code block (triple backtick, no language tag)
- Use plain Unicode in pseudocode: `←` for assignment, `~` for sampling, `‖...‖` for norms, `#` for inline comments
- Use ASCII-friendly math where possible: `x̂_0`, `σ_t²`, `Δt`, `ε ~ N(0,I)`
- No LaTeX inside the code block
- Step numbers (`1.`, `2.`, ...) are mandatory for the main loop body
- Gradient tracking must be noted explicitly: `# with grad` or `# no grad` on the relevant lines
- If the paper introduces a named variant (Fast, Flash, etc.), include it as a clearly separated block at the end of the same code block
- The algorithm must be self-contained: someone who only reads the pseudocode should be able to implement the method

#### Reference Implementation (VeRL-Omni)

If the method's **loss** is registered in VeRL-Omni's [`diffusion_algos.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/diffusion/diffusion_algos.py) (register names: `flow_grpo`, `dance_grpo`, `flow_dppo`, `grpo_guard`, `dpo`, `diffusion_nft`), add a short subsection `## Reference Implementation (VeRL-Omni)` **after** `## Algorithm` containing a **condensed** functional form of the registered loss (≈ 5–10 lines), and link the upstream file. Do **not** transcribe the full class — keep only the core math so a reader can map the page's objective onto the runnable code. Keep the rollout/sampling pseudocode in `## Algorithm` (that part is not in the registry).

Three rules for the condensed form:
- **FlowGRPO-family extensions** (GRPO-Guard, FlowDPPO, …) — present the snippet as a **diff against `FlowGRPOLoss`**: keep the shared body verbatim and mark every changed line with a trailing `# <<<` comment naming what changed, so the delta is obvious at a glance.
- **Sampler / scheduler improvements** that are *not* a registered loss (e.g. CPS = `sde_type="cps"`) — show the relevant **scheduler branch diff** (what changes when the flag is enabled vs. the default), link the scheduler file + the example that enables it, and state which config flag selects it.
- **Methods with non-trivial reward preprocessing** (e.g. DiffusionNFT's reward → `reward_prob`) — include the reward/advantage preparation step too, not just the loss, so the snippet is runnable end-to-end in spirit.

Example condensed form:

```python
@register_diffusion_loss("flow_grpo")   # also registered for "dance_grpo"
def loss_flow_grpo(old_lp, lp, adv, cfg):
    c = cfg.diffusion_loss
    adv = clamp(adv, -c.adv_clip_max, c.adv_clip_max)
    ratio = exp(lp - old_lp)
    unclipped = -adv * ratio
    clipped   = -adv * clamp(ratio, 1 - c.clip_ratio, 1 + c.clip_ratio)
    return mean(max(unclipped, clipped))         # PPO-clip
```

---

### 4.9 Limitations Section

```markdown
## Limitations
```

**Two formats** — choose based on whether the limitation is addressed in this repo:

**Format A — table** (use when at least one limitation is addressed by another paper in the repo):
```markdown
| Problem | Addressed by |
|---|---|
| [concise problem description] | [Paper](relative_link.md) |
| [problem with no known solution in this repo] | — |
```

**Format B — bullet list** (use when no limitations are addressed by other papers in the repo):
```markdown
- [Limitation sentence]
- [Limitation sentence]
```

Rules:
- Every major limitation from the paper's own discussion section should appear
- For Format A: the "Addressed by" column should use the short name as a link; use `—` if not addressed in the repo
- Limitations should be specific (a failure mode that can be reproduced or measured), not vague ("may be unstable")
- Do not include "future work" items — only current limitations

---

## 5. Notation Rules for New Symbols

When a paper introduces a symbol not in NOTATION.md, follow this decision tree:

```
Is the symbol used in more than one paper, or likely to appear in future papers?
  YES → Add it to NOTATION.md under the appropriate section
  NO  → Define it only in the notation header of this file

Does the new symbol conflict with an existing NOTATION.md symbol?
  YES → Use the NOTATION.md symbol in the file body;
        note the paper's choice in the notation header: "[paper uses X for our Y]"
  NO  → Introduce the new symbol with definition on first use in the file body
```

**When updating NOTATION.md**:
- Add to the existing section that best fits the concept (don't create new top-level sections without strong reason)
- Follow the existing table format
- Add a short definition and the papers that use the symbol

---

## 6. Cross-Referencing Rules

**Within papers/policy_gradient/**: use bare filename: `[CPS](cps.md)`

**Across directories**: use relative path: `[DGPO](../direct_preference/dgpo.md)`

**External papers with no file in this repo**: use arXiv link inline — `Diffusion-DPO ([2311.12908](https://arxiv.org/abs/2311.12908))`

**Never** link to `foundations/` — that directory has been removed.

---

## 7. Files to Update After Adding a Paper

When a new dedicated paper file is added to `policy_gradient/` or `direct_preference/`, update these files in order:

### 7.1 `papers/INDEX.md` — Master Table

Add one row to the master table, sorted by submission date:

```markdown
| **ShortName** | Full title | [NNNN.NNNNN](https://arxiv.org/abs/NNNN.NNNNN) | YYYY-MM-DD | Venue | [paradigm/filename.md](paradigm/filename.md) |
```

Then update the **Citation Graph** section:
- Add a block for the new paper listing which papers it cites and which cite it
- Update the blocks of papers it cites: add `→ NewPaper` to their citation edges
- Update the paradigm lineage diagram if the paper fits into an existing branch

### 7.2 `papers/academia.md` — Remove Existing Entry if Present

If the paper was previously tracked as an academia.md entry (before it got its own file), remove that entry from academia.md. Keep the master index table entry in INDEX.md.

### 7.3 `papers/READING_GUIDE.md` — Narrative Placement

If the paper is a direct successor to an existing paper (addresses a specific limitation, is explicitly positioned against a prior paper in the chain):
- Add a step to **Part I** (policy-gradient chain) or **Part II** (direct-preference branch) with the same structure as existing steps:
  - What the predecessor left open
  - What this paper specifically changes
  - The "keeps / changes" summary

If the paper is a parallel variant or a 2026+ advance without a clear chain position, add a brief mention to **Part III** under an appropriate subsection.

Update the **Quick Reference table** at the bottom of READING_GUIDE.md with a row for the new paper.

### 7.4 `README.md` — Directory Layout

If the new file is in `papers/policy_gradient/` or `papers/direct_preference/`, add one line to the directory tree in the layout section:

```
│   ├── new_paper.md          ← ShortName (Mon YYYY) — one-line description
```

Keep the directory tree sorted by submission date within each subdirectory.

### 7.5 Update `**Cited by**` Fields

Go back to any paper that cites the new one (listed in the new paper's **Cites** field) and add the new paper's short name to their **Cited by** metadata field.

---

## 8. Adding an `academia.md` Entry

When a paper does not warrant a dedicated file, add a short entry to `academia.md`:

### 8.1 Master index table row

Add one row to the master index table at the top of `academia.md` (sorted by date):

```markdown
| YYYY-MM | [NNNN.NNNNN](https://arxiv.org/abs/NNNN.NNNNN) | ShortName | Policy Gradient/Direct Preference | Key problem in 8 words |
```

### 8.2 Detailed entry

Add under the appropriate section heading (`## Policy Gradient Paradigm Advances` or `## Direct Preference Paradigm Advances`):

```markdown
### ShortName — `NNNN.NNNNN` · Mon YYYY

**GitHub**: URL or omit if none

**Problem**: [One sentence: the specific failure mode or gap.]

**Approach**: [Two to four sentences. State the mechanism and its key parameters. Include the most important equation if it fits.]

$$\text{key formula if applicable}$$

[Optional: results in one sentence if notable.]
```

---

## 9. Scope of New Papers for Models Section

When adding an industry model (not an academic paper) to `models.md`, different rules apply:
- The model must have a public technical report or paper documenting its RL training
- The entry should follow the existing per-company format in `models.md`
- Link to academic papers in this repo using their file paths; link to external papers using arXiv URLs

---

## 10. Pre-Commit Checklist

Before finalising any addition, verify:

- [ ] **Scope**: paper satisfies all three criteria in §1
- [ ] **Paradigm**: policy-gradient vs. direct-preference classification is correct (check the test question in §2)
- [ ] **Notation**: every symbol in the file matches NOTATION.md; local deviations declared in header
- [ ] **Boxed objective**: the main training loss is in `$$\boxed{...}$$`
- [ ] **Algorithm**: pseudocode is self-contained; gradient tracking annotated; variant blocks present if applicable
- [ ] **Cross-references**: all `[Name](link.md)` links use correct relative paths and point to existing files
- [ ] **Metadata**: arXiv link works; GitHub link verified or `—`; Submitted date is first submission date
- [ ] **INDEX.md**: new row added; citation edges updated for cites and cited-by
- [ ] **Cited-by fields**: existing paper files updated with the new paper's short name
- [ ] **README.md**: directory tree updated
- [ ] **READING_GUIDE.md**: new paper placed in the narrative or noted in the Quick Reference table

---

## 11. Example File Skeleton

Copy this skeleton and fill in each `[...]` field:

```markdown
# [ShortName] — [Full Paper Title]

> Notation: follows [NOTATION.md](../NOTATION.md)[, §N (section name)]. [Local symbols: $x$ — description.]

| Field | Value |
|---|---|
| **arXiv** | [[NNNN.NNNNN](https://arxiv.org/abs/NNNN.NNNNN)] |
| **Submitted** | [YYYY-MM-DD] |
| **Venue** | [Venue or — (preprint)] |
| **Authors** | [Full author list] |
| **GitHub** | [URL or —] |
| **Paradigm** | **[Policy Gradient/Direct Preference]** — [one-clause description] |
| **Cites** | [Short names of relevant cited papers] |
| **Cited by** | [Leave — if new; fill after later papers are added] |

---

## Context

[One paragraph. What predecessors established. What gap this paper fills. Links to prerequisite files.]

---

## Problem 1 — [Descriptive title of the bottleneck]

**Issue**: [What breaks. Quantify if possible.]

**Idea**: [What is proposed, as a mechanism.]

**Why this works**: [The structural reason — theorem, duality, or property.]

[Derivation / equations here.]

---

## Problem 2 — [If applicable]

...

---

## Training Objective

[One sentence connecting problems to the objective.]

$$\boxed{
\mathcal{L}_\text{[Name]}(\theta) = ...
}$$

where [symbol definitions].

---

## Algorithm

```
Input: [hyperparameters]
Repeat:
  1. [step]
  2. [step]
```

---

## Limitations

| Problem | Addressed by |
|---|---|
| [limitation] | [Paper](link.md) or — |
```

---

## 12. Quick Reference: Repository Layout

```
diffusion-rl-survey/
├── README.md
├── INTEGRATION_GUIDE.md     ← this file
├── models.md
└── papers/
    ├── INDEX.md             ← update for every new paper
    ├── NOTATION.md          ← update when adding new shared symbols
    ├── READING_GUIDE.md     ← update narrative when a new key paper is added
    ├── academia.md          ← short entries for papers without dedicated files
    ├── prerequisites/       ← GRPO + flow-matching primers
    ├── policy_gradient/     ← objective = PPO-clip policy gradient over the trajectory
    │   └── *.md
    └── direct_preference/   ← objective = preference / MSE loss on final samples
        └── *.md
```
