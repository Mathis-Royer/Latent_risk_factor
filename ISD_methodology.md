# Implementation Specification Document (ISD) — General Methodology

## For multi-agent orchestration of technically dense coupled projects

**Version 1.0**

---

## Table of Contents

- [Introduction](#introduction)
- [Part A — User: irreducible substrate](#part-a--user-irreducible-substrate)
  - [A.1 Environmental prerequisites](#a1-environmental-prerequisites)
  - [A.2 Production of the Technical Vision Document (DVT)](#a2-production-of-the-technical-vision-document-dvt)
  - [A.3 Identification of domain-specific critical invariants](#a3-identification-of-domain-specific-critical-invariants)
  - [A.4 Definition of global acceptance criteria](#a4-definition-of-global-acceptance-criteria)
  - [A.5 Part A Deliverables — Checklist](#a5-part-a-deliverables--checklist)
- [Part B — Supervised AI agent: structural compilation](#part-b--supervised-ai-agent-structural-compilation)
  - [B.1 DVT analysis and topology extraction](#b1-dvt-analysis-and-topology-extraction)
  - [B.2 Modular decomposition](#b2-modular-decomposition)
  - [B.3 Dependency graph construction](#b3-dependency-graph-construction)
  - [B.4 Sequential / parallel classification](#b4-sequential--parallel-classification)
  - [B.5 Context density estimation per module](#b5-context-density-estimation-per-module)
  - [B.6 Production of inter-module assertions](#b6-production-of-inter-module-assertions)
  - [B.7 Structured ISD generation](#b7-structured-isd-generation)
  - [B.8 Human validation and iteration](#b8-human-validation-and-iteration)
  - [B.9 Part B Deliverables — Checklist](#b9-part-b-deliverables--checklist)
- [Part C — Autonomous agents: orchestrated execution](#part-c--autonomous-agents-orchestrated-execution)
  - [C.1 Orchestration architecture](#c1-orchestration-architecture)
  - [C.2 Project CLAUDE.md configuration](#c2-project-claudemd-configuration)
  - [C.3 Structure of an ISD section consumable by an agent](#c3-structure-of-an-isd-section-consumable-by-an-agent)
  - [C.4 Builder-validator protocol](#c4-builder-validator-protocol)
  - [C.5 Execution phase management](#c5-execution-phase-management)
  - [C.6 Inter-agent communication conventions](#c6-inter-agent-communication-conventions)
  - [C.7 Completion criteria and error handling](#c7-completion-criteria-and-error-handling)
  - [C.8 Operational templates](#c8-operational-templates)
  - [C.9 Anti-patterns and diagnostics](#c9-anti-patterns-and-diagnostics)
  - [C.10 Part C Deliverables — Checklist](#c10-part-c-deliverables--checklist)
- [Appendix — Glossary](#appendix--glossary)

---

## Introduction

### Problem addressed

Technically dense coupled projects — quantitative research pipelines, scientific modeling systems, end-to-end ML architectures — exhibit a structural characteristic that makes them resistant to standard multi-agent orchestration approaches: design decisions at one point in the pipeline have cascading, often non-linear, ramifications on distant components. A modification to the loss function affects auto-pruning, which conditions the number of active factors, which constrains the downstream covariance estimation, which determines the behavior of the portfolio solver.

Existing approaches — Ralph (autonomous loop on PRD JSON), native Claude Code agent teams (default parallelization), subagents (focused delegation) — provide powerful orchestration primitives but none solves the **compilation** problem: transforming a dense methodological document into self-sufficient specifications, correctly segmented, and annotated for orchestration.

### What this document formalizes

This document defines a three-phase method for producing an **Implementation Specification Document (ISD)** — an intermediate artifact between the technical vision document (written by the human) and the code (produced by the agents). The ISD is the product of a cross-analysis between the domain's dependency structure and the operational constraints of the agents (context window, absence of inter-session memory, limited communication).

### Scope and limitations

This methodology applies to projects satisfying three conditions: (1) sufficient technical complexity to justify multi-agent orchestration (> 2,000 expected lines of code, > 5 interdependent modules), (2) existence of a detailed methodological or specification document, (3) inter-module coupling that prevents a naive decomposition into independent tasks.

It does not apply to CRUD projects, refactoring of isolated modules, or exploratory prototypes where rapid iteration in a single session is more efficient.

---

## Part A — User: irreducible substrate

This part describes what the human must produce **before any AI agent intervention**. These elements constitute the irreducible substrate of the project: domain-specific knowledge, fundamental architectural choices, and environmental configuration that only the domain expert can provide.

### A.1 Environmental prerequisites

#### A.1.1 Claude Code installation and configuration

The execution environment must be prepared before the ISD is produced. The following steps are imperative and non-delegable to an agent.

**Claude Code.** Ensure that Claude Code is installed and functional. Verify the version (`claude --version`) and the availability of necessary tools (bash, read, write, edit, glob, grep). Configure the default model according to the project's complexity needs.

**Agent Teams (experimental).** If the project requires inter-agent parallelization (see B.4 for decision criteria), enable the feature:

```json
// settings.json or environment variable
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

Note: Agent Teams is experimental. Known limitations concern session recovery, task coordination, and shutdown behavior. For critical projects, plan for manual checkpoints.

**Terminal.** Split-pane mode (simultaneous teammate visualization) requires tmux or iTerm2. In-process mode (default) works in any terminal. The VS Code integrated terminal, Windows Terminal, and Ghostty do not support split-pane.

**Git.** Initialize the project repository before the start of orchestration. Agents will use Git as a persistence and rollback mechanism. Configure working branches according to the planned pattern (one branch per phase, per module, or per agent — to be decided in B.4).

#### A.1.2 Project structure

Create the minimal directory structure:

```
project_root/
├── .claude/
│   ├── settings.json          # Claude Code configuration
│   ├── agents/                # Subagent definitions (Part C)
│   └── commands/              # Custom slash commands
├── CLAUDE.md                  # Global project context (Part C)
├── docs/
│   ├── dvt.md                 # Technical Vision Document (Part A)
│   ├── isd/                   # ISD sections (Part B)
│   │   ├── 00_global.md       # Global invariants and conventions
│   │   ├── 01_module_xxx.md   # One section per module
│   │   └── ...
│   └── assertions/            # Inter-module assertions (Part B)
│       └── contracts.yaml     # Formalized interface contracts
├── src/                       # Code produced by agents
│   ├── module_xxx/
│   └── ...
├── tests/                     # Unit and integration tests
│   ├── unit/
│   └── integration/
└── scripts/                   # Utility scripts
```

#### A.1.3 Dependency management

The user must specify the target execution environment: Python version (or other language), system dependencies, required libraries and their versions. This information is critical because agents must not make autonomous dependency decisions — a library choice (e.g., PyTorch vs JAX, CVXPY vs SciPy) has architectural ramifications that only the domain expert can arbitrate.

Formalize in a specification file (e.g., `requirements.txt`, `pyproject.toml`, `environment.yml`) **before** the start of orchestration.

### A.2 Production of the Technical Vision Document (DVT)

The DVT is the source document from which the ISD will be compiled. It is the only document that the user must write entirely. Its quality directly determines the quality of the ISD and, consequently, the quality of the code produced by the agents.

#### A.2.1 Required content

The DVT must contain, at minimum:

**Vision and objective.** The project's intention formulated in verifiable terms. Not "build a good model" but "produce a model whose metric X on data Y exceeds threshold Z". The objective must be precise enough for an agent to determine whether the project is complete.

**Global architecture.** A description of the end-to-end pipeline, including data flow between components. A diagram (even text-based) is essential. The architecture must specify entry points (source data), successive transformations, and outputs (deliverables).

**Technical specifications per component.** For each pipeline component:
- Expected inputs (types, shapes, naming conventions, units).
- Applied transformations (complete mathematical formulas, algorithms, pseudo-code).
- Produced outputs (same specifications as inputs).
- Configurable parameters (default values, valid ranges, justification).
- Edge cases and their handling.

**Design choices and justifications.** Each non-trivial decision must be accompanied by its justification. This information is critical for agents: it allows them to distinguish what is modifiable (implementation choices) from what is constrained (intentional design choices). An agent lacking this information risks "correcting" a deliberate choice believing it is fixing a bug.

**Validation protocol.** How the project will be evaluated: metrics, experimental protocol (e.g., walk-forward), benchmarks, success/failure criteria. This protocol must be detailed enough to be implementable without ambiguity.

**References.** Any article, documentation, or external source that agents might need. Agents do not necessarily have access to these sources — relevant elements must be cited and summarized in the DVT.

#### A.2.2 Required level of detail

The DVT's level of detail conditions agent autonomy. The guiding rule:

> **Any element that, if poorly implemented, silently invalidates the downstream pipeline, must be specified at a level of detail that leaves no room for interpretation.**

Concretely, this means:
- Mathematical formulas must be written in explicit notation, with tensor dimensions and indexing conventions.
- Constants must have numerical values, not qualitative descriptions ("a small number" → $10^{-8}$).
- Edge cases must be handled explicitly ("if the denominator is < $10^{-8}$, clamp to $10^{-8}$").
- Interactions between components must be documented ("modifying X affects Y via mechanism Z").

#### A.2.3 Completeness indicators

The DVT is sufficiently complete when:

1. **Naive expert reader test.** A competent engineer in the technical domain (but ignorant of the project) could implement each component without asking clarification questions.
2. **Alternative implementation test.** Two independent implementations produced from the DVT would yield numerically identical results (up to floating-point errors).
3. **Deletion test.** No paragraph can be removed without an agent being able to produce an incorrect implementation.

### A.3 Identification of domain-specific critical invariants

Critical invariants are system properties that must be true at all times and whose violation is not detectable by standard unit tests. They result from the user's domain-specific knowledge and cannot be automatically derived from the DVT.

#### A.3.1 Invariant categories

**Mathematical invariants.** Algebraic properties that must be respected in the code. Examples: a covariance matrix must be positive semi-definite; a normalization factor must be present for two terms of a loss to be comparable; a sign convention must be consistent across components.

**Inter-module consistency invariants.** Properties that span the interface between two modules and are verifiable only by joint inspection. Example: the output tensor of module A has shape $(n, K)$ that must match the input expected by module B, but $K$ is determined dynamically by module A (auto-pruning) and module B must adapt.

**Convention invariants.** Convention choices that, if they differ between modules, produce numerically incorrect results without explicit error. Examples: log vs arithmetic returns; covariance per observation vs per element; 0-based vs 1-based indices.

**Financial safety invariants.** Specific to quantitative projects: absence of look-ahead (no future data in calculations), point-in-time universe construction, correct handling of delistings, dividend and split management.

#### A.3.2 Formalization

Each invariant must be formalized according to the following template:

```yaml
invariant:
  id: INV-001
  category: mathematical | inter_module | convention | safety
  description: "The factor D = T × F must appear in the numerator
                of the reconstruction loss to balance with the KL"
  modules_affected: [loss_function, training, monitoring]
  violation_consequence: "Posterior collapse — all latent dimensions
                         converge to the prior, the model becomes useless"
  detection_method: "Verify that AU > 0.15 × K after training.
                     If AU < 5, the factor D is probably missing."
  test_assertion: |
    # The reconstruction loss must be multiplied by D/(2*sigma_sq)
    # and not by 1/(2*sigma_sq)
    assert loss_recon_coeff == (T * F) / (2 * sigma_sq)
```

#### A.3.3 Extraction process

Identifying invariants is a deliberate exercise, not a byproduct of writing the DVT. The user must, for each pipeline component:

1. List the implicit assumptions that the component makes about its inputs.
2. List the properties that the component guarantees about its outputs.
3. Identify the properties that cross module boundaries (i.e., assumptions of one module that depend on guarantees of another).
4. For each inter-module property, ask: "If one agent correctly implements module A and another agent correctly implements module B, but with different conventions, does the pipeline silently produce incorrect results?" If yes, it is a critical invariant.

### A.4 Definition of global acceptance criteria

Acceptance criteria define the conditions under which the project is considered complete. They differ from performance metrics in that they include structural and code quality criteria, not just numerical results.

#### A.4.1 Structural criteria

- **Test coverage.** Minimum coverage threshold (e.g., > 80% of lines, > 90% of critical branches).
- **Reproducibility.** Given a fixed seed, two runs of the complete pipeline produce identical results.
- **Documentation.** Each public function has a docstring specifying inputs, outputs, and edge case behavior.
- **Invariant compliance.** All invariants from A.3 are verified by automated tests.

#### A.4.2 Functional criteria

- **Completeness.** All DVT components are implemented and integrated.
- **End-to-end execution.** The complete pipeline runs without errors on test data.
- **Benchmarks.** Performance metrics meet the thresholds defined in the DVT.

#### A.4.3 Code quality criteria

- **No dead code.** No unused functions, unreachable branches, or unresolved TODO comments.
- **Modularity.** Each module is independently importable and testable.
- **Performance.** The execution time constraints specified in the DVT are met.

### A.5 Part A Deliverables — Checklist

At the end of Part A, the user must have produced:

- [ ] Claude Code environment configured and functional
- [ ] Agent Teams enabled (if parallelization required)
- [ ] Directory structure created
- [ ] Dependencies specified (`requirements.txt` or equivalent)
- [ ] Complete DVT (docs/dvt.md) satisfying the three completeness tests
- [ ] Critical invariants formalized (docs/assertions/invariants.yaml)
- [ ] Global acceptance criteria defined
- [ ] Git repository initialized with an initial commit containing all deliverables

---

## Part B — Supervised AI agent: structural compilation

This part describes the analytical work that the AI agent performs under the direct supervision of the user. The objective is to **compile** the DVT into a structured ISD — that is, to transform a dense and coupled methodological document into self-sufficient specifications, correctly segmented, and annotated for multi-agent orchestration.

Each step of this part produces an intermediate artifact submitted for user validation before proceeding to the next step. The agent must not proceed to step $n+1$ without explicit validation of step $n$.

### B.1 DVT analysis and topology extraction

#### B.1.1 Objective

Identify the **logical structure** of the project as it emerges from the DVT: functional components, their relationships, data flows, and coupling points. This analysis is distinct from modular decomposition (B.2) — it concerns the problem's topology, not the code segmentation.

#### B.1.2 Procedure

The agent performs an analytical reading of the DVT and produces:

**Functional component inventory.** Each component is a logical unit that transforms inputs into outputs. It does not necessarily correspond to a code module — a component may be spread across multiple modules, or multiple components may coexist within a single module.

**Data flow graph.** A DAG (Directed Acyclic Graph) where nodes are components and edges represent data flows. Each edge is annotated with the type of data transmitted (tensor shape, type, convention).

**Coupling matrix.** For each pair of components $(i, j)$, estimate the degree of coupling on a scale:
- **0 — Independent.** No interaction. Components can be implemented without knowledge of each other.
- **1 — Interface.** Coupling only through the input/output interface. Specifying the interface contract is sufficient.
- **2 — Parametric.** One component produces a parameter that affects the behavior (not just the data) of the other. Example: the number of active dimensions (AU) determined by the VAE conditions the size of matrices in the risk model.
- **3 — Semantic.** Components share implicit conventions whose violation produces numerically incorrect results without error. Example: the normalization convention (per-element vs per-window) must be identical between the loss function and monitoring.
- **4 — Mathematical.** Components are linked by a mathematical identity whose violation invalidates the convergence proof, theoretical justification, or expected system behavior. Example: the factor $D$ in the loss links reconstruction and KL in a way that guarantees self-regulated balancing.

**Critical coupling points.** Pairs with coupling degree >= 3 are critical coupling points. The agent lists them with a description of the invariant that links them. These points will determine sequentiality constraints in B.4.

#### B.1.3 Output format

```yaml
topology:
  components:
    - id: COMP-001
      name: "Data preprocessing"
      dvt_sections: ["4.2"]
      inputs: ["raw_prices: DataFrame(n_stocks × n_dates)"]
      outputs: ["windows: Tensor(n_windows × T × F), crisis_labels: Array(n_windows)"]

  data_flows:
    - from: COMP-001
      to: COMP-002
      data: "windows: Tensor(n_windows × T × F)"
      coupling_degree: 1

  critical_couplings:
    - components: [COMP-003, COMP-005]
      degree: 4
      invariant: "INV-001 — factor D in the loss"
      description: "The loss function and monitoring must use
                    the same convention D = T × F for σ² to be interpretable"
```

#### B.1.4 User validation point

The user verifies:
- All DVT components are represented.
- Coupling degrees are correct (the agent tends to underestimate semantic and mathematical coupling — the domain expert is the only reliable judge).
- Critical coupling points have no omissions.

### B.2 Modular decomposition

#### B.2.1 Objective

Transform the functional topology (B.1) into a decomposition into **code modules** — units that are implementable, unit-testable, and assignable to an agent.

#### B.2.2 Decomposition principles

Modular decomposition simultaneously optimizes three criteria in tension:

**Minimization of inter-module coupling.** Module boundaries must pass through the minimal coupling edges (degree 0 or 1) of the topological graph. Two components with coupling degree >= 3 must **never** be separated into different modules, unless the coupling can be entirely captured by a formalized interface contract.

**Maximization of intra-module cohesion.** A module must perform a logically complete function. The test: a developer can understand the module without reading others, based solely on interface contracts.

**Compliance with the context constraint.** Each module must be specifiable (formulas, pseudo-code, tests) in a volume less than ~60% of an agent's context window (reserving ~40% for produced code and interactions). For Claude Code with Opus 4.6 (1M tokens), this represents ~600K tokens of specification — generally non-binding. For smaller models or reduced contexts, this constraint becomes active and forces finer granularity.

#### B.2.3 Procedure

1. **Initial partitioning.** Group functional components (B.1) into clusters of coupling >= 2, using the coupling graph. Each cluster becomes a candidate module.

2. **Consistency verification.** For each candidate module, verify that it performs a function understandable in isolation. If a module contains functionally disjoint components (grouped only by coupling), consider splitting while preserving coupling interfaces.

3. **Size verification.** Estimate the specification volume of each module (see B.5). If a module exceeds the context constraint, split it into sub-modules — but only along coupling boundaries <= 2. If no such boundary exists, the module is **irreducible** and must be implemented by an agent in a single session with full context.

4. **Ordering.** Define an implementation order that respects dependencies: a module can only be implemented after those it depends on (incoming coupling degree >= 2).

#### B.2.4 Output format

```yaml
modules:
  - id: MOD-001
    name: "data_pipeline"
    components: [COMP-001, COMP-002]
    dependencies: []
    estimated_context_density: "low"
    parallelizable: true

  - id: MOD-003
    name: "loss_function"
    components: [COMP-005, COMP-006, COMP-007]
    dependencies: [MOD-002]
    estimated_context_density: "high"
    parallelizable: false
    irreducible: true
    reason: "Mathematical coupling degree 4 between the three components"
```

### B.3 Dependency graph construction

#### B.3.1 Objective

Formalize the inter-module dependency DAG in a form directly transposable into a task list for orchestration (Claude Code Agent Teams or subagents).

#### B.3.2 Dependency types

**Data dependency (D).** Module B consumes an output of module A. Module B cannot be tested without a stub or mock of A's output.

**Interface dependency (I).** Module B must know A's output interface (types, shapes) to define its input interface, but does not consume A's actual data during development. Development can be parallelized if the interface is frozen.

**Code dependency (C).** Module B imports and uses code from module A (utility functions, shared classes). Module A must be implemented and stable before B.

**Validation dependency (V).** Module B can only be validated jointly with A (integration tests). Implementation can be parallel, but validation is sequential.

#### B.3.3 DAG formalization

```yaml
dependency_graph:
  edges:
    - from: MOD-001  # data_pipeline
      to: MOD-003    # loss_function
      type: D        # data dependency
      interface: "windows: Tensor(n × T × F)"
      blocking: true

    - from: MOD-002  # vae_architecture
      to: MOD-003    # loss_function
      type: C        # code dependency
      interface: "VAEModel class with forward() method"
      blocking: true

    - from: MOD-001  # data_pipeline
      to: MOD-002    # vae_architecture
      type: I        # interface dependency only
      interface: "input_shape: (T, F)"
      blocking: false  # parallelizable if interface is frozen

  critical_path:
    - [MOD-001, MOD-003, MOD-004, MOD-005, MOD-006, MOD-007]

  parallelizable_groups:
    - [MOD-001, MOD-002, MOD-008]  # no mutual dependency
    - [MOD-009, MOD-010, MOD-011]  # independent benchmarks
```

#### B.3.4 User validation point

The user verifies that:
- The critical path is correct (no missing dependency that would create a blocking situation during execution).
- Type I dependencies are genuinely non-blocking (the interface is stable enough to be frozen).
- Parallelizable groups are effectively independent.

### B.4 Sequential / parallel classification

#### B.4.1 Objective

For each module, determine the optimal execution mode: lead session, subagent, or agent team teammate. This classification is the product of cross-analysis between the dependency graph (B.3) and the operational characteristics of each orchestration primitive.

#### B.4.2 Decision matrix

| Criterion | Lead session | Subagent | Agent Team teammate |
|-----------|-------------|----------|---------------------|
| Coupling with other modules | Degree >= 3 with current module | Degree <= 2, sequential | Degree <= 1, parallel |
| Context density | Very high (> 70% window) | Medium to high | Low to medium |
| Need for in-progress human validation | Yes (design choices) | No (complete specification) | No |
| Required inter-agent communication | N/A | No (reports to lead) | Yes (sharing findings) |
| Estimated duration | Variable | < 1h of agent context | < 1h of agent context |
| Independence | N/A | High (after dependencies resolved) | Very high |
| Token cost | 1x | 1x per subagent | 1x per teammate (additive) |

#### B.4.3 Classification heuristics

**Lead session** — use for:
- Irreducible modules (B.2.3) whose specification exceeds a subagent's capacity.
- Modules requiring real-time design decisions (iteration with the user).
- Final integration and end-to-end tests.

**Subagent** — use for:
- Well-specified modules with resolved dependencies (outputs from previous modules available).
- Refactoring, documentation, or test writing tasks after the main implementation.
- The builder-validator pattern (one subagent codes, another validates — see C.4).

**Agent Team teammate** — use for:
- Parallelizable module groups identified in B.3.
- Independent modules of comparable size (avoid imbalance where one teammate finishes well before the others).
- Exploration phases (comparing two implementation approaches in parallel, then selecting the best).

#### B.4.4 Module annotation

Each module receives an orchestration annotation:

```yaml
modules:
  - id: MOD-003
    name: "loss_function"
    orchestration:
      mode: "lead_session"
      justification: "Mathematical coupling degree 4; context density > 70%;
                      interaction with 3 modes (P/F/A) requiring global understanding"
      phase: 2

  - id: MOD-001
    name: "data_pipeline"
    orchestration:
      mode: "agent_team_teammate"
      justification: "Independent; interface coupling only;
                      parallelizable with MOD-002 and MOD-008"
      phase: 1
      team: "infrastructure"
```

#### B.4.5 Execution phase definition

Modules are grouped into sequential **phases**, each phase containing modules executable in parallel:

```
Phase 1 (parallel): MOD-001, MOD-002, MOD-008
    ↓ synchronization
Phase 2 (sequential): MOD-003 → MOD-004 → MOD-005 → MOD-006 → MOD-007
    ↓ synchronization
Phase 3 (parallel): MOD-009, MOD-010, MOD-011, MOD-012
    ↓ synchronization
Phase 4 (integration): MOD-013
```

### B.5 Context density estimation per module

#### B.5.1 Objective

Estimate, for each module, the specification volume that the agent will need to maintain in context during implementation. This estimation determines whether a module can be assigned to a subagent (bounded context) or requires a lead session (full context).

#### B.5.2 Estimation method

For each module, account for:

**Mathematical specifications.** Number of formulas, edge cases, parameters. Weight by complexity: a simple formula (1 line) = 1 unit; a formula with conditional cases, clamping, interactions = 3–5 units.

**Input/output interfaces.** Number of distinct data types in input and output, with their conventions (shapes, units, normalizations).

**Applicable invariants.** Number of invariants (A.3) that apply to this module. Each invariant must be in context during implementation.

**Existing code to know.** If the module depends on code from previous modules (C dependency), the interfaces of that code must be in context.

**Volumetric estimation.** As a first approximation:

$$V_{\text{context}}(\text{module}) \approx V_{\text{spec}} + V_{\text{interfaces}} + V_{\text{invariants}} + V_{\text{code\_dep}}$$

Where each $V$ is estimated in tokens (~0.75 words per token in technical English, ~4 characters per token in Python code).

#### B.5.3 Thresholds

| Density | Estimated volume | Recommended mode |
|---------|-----------------|-----------------|
| Low | < 10K tokens | Subagent or teammate |
| Medium | 10K – 50K tokens | Subagent preferred |
| High | 50K – 200K tokens | Lead session or subagent with extended context |
| Very high | > 200K tokens | Lead session mandatory; consider splitting the module |

### B.6 Production of inter-module assertions

#### B.6.1 Objective

Produce a set of formal assertions that serve as a **contract** between modules. These assertions are the primary mechanism for detecting inconsistency when modules are implemented by different agents.

#### B.6.2 Assertion types

**Shape assertions.** Verification of tensor dimensions at interfaces.

```python
# Assertion: the encoder output has the correct shape
assert encoder_output.shape == (n_stocks, K), \
    f"Encoder output shape {encoder_output.shape} != expected ({n_stocks}, {K})"
```

**Type and range assertions.** Verification that values are within expected ranges.

```python
# Assertion: σ² is scalar, positive, and within the clamp range
assert sigma_sq.ndim == 0, "σ² must be scalar"
assert 1e-4 <= sigma_sq.item() <= 10, f"σ² = {sigma_sq.item()} outside [1e-4, 10]"
```

**Convention assertions.** Verification that conventions are consistent between modules.

```python
# Assertion: returns are in log (not arithmetic)
# Indirect verification: log-returns sum ≈ period return
assert abs(returns.sum() - np.log(prices[-1]/prices[0])) < 1e-10, \
    "Returns are not log-returns — sum does not equal log(P_T/P_0)"
```

**Mathematical consistency assertions.** Verification of mathematical identities across modules.

```python
# Assertion: the factorial covariance reconstructs the asset covariance
Sigma_reconstructed = B_A @ Sigma_z @ B_A.T + D_eps
Sigma_sample = np.cov(returns.T)
relative_error = np.linalg.norm(Sigma_reconstructed - Sigma_sample) / np.linalg.norm(Sigma_sample)
assert relative_error < 0.5, f"Factor model reconstruction error {relative_error:.2f} > 0.5"
```

**Temporal integrity assertions.** Specific to financial projects: verification of the absence of look-ahead.

```python
# Assertion: no future data in the calculation
assert all(train_dates < embargo_start), "Training data leaks past embargo"
assert all(test_dates > embargo_end), "Test data starts before embargo ends"
```

#### B.6.3 Organization

Assertions are organized into three levels:

1. **Unit assertions** — internal to a module, included in the module's tests.
2. **Interface assertions** — verified at junction points between modules, included in integration tests.
3. **Global assertions** — end-to-end pipeline properties, included in system tests.

Each assertion references the invariant (A.3) it verifies and the modules it concerns.

### B.7 Structured ISD generation

#### B.7.1 Objective

Produce the final document: the ISD, which is the compilation of all preceding work into a format directly consumable by agents.

#### B.7.2 ISD structure

The ISD is a set of Markdown files, one per module plus a global file:

```
docs/isd/
├── 00_global.md           # Conventions, invariants, glossary
├── 01_data_pipeline.md    # Module MOD-001
├── 02_vae_architecture.md # Module MOD-002
├── ...
└── XX_integration.md      # Integration module
```

**The global file** (`00_global.md`) contains:
- Naming, indexing, and notation conventions.
- Glossary of terms and symbols.
- Complete list of invariants (extracted from A.3).
- Dependency graph (extracted from B.3).
- Phase plan (extracted from B.4).
- Global acceptance criteria (extracted from A.4).

**Each module file** contains the elements detailed in C.3 (structure of an ISD section consumable by an agent). The crucial point: each file is **self-sufficient** — an agent reading only this file and the global file must have all the information necessary to implement the module.

#### B.7.3 Self-sufficiency rule

Self-sufficiency is the most important property of the ISD and the most difficult to achieve. It means:

- **No references to the DVT.** If the DVT says "see Section 4.7 for the rescaling formula", the ISD must reproduce the formula in full in the module section that uses it.
- **No cross-module references.** If module B depends on the output of module A, module B's ISD must specify the shape, type, and conventions of this output — even if it is redundant with module A's ISD. Redundancy is the price of self-sufficiency.
- **Complete formulas.** Each mathematical formula is written with locally defined variables, units, indexing conventions, and edge cases.

The cost of redundancy (larger text volume) is far less than the cost of an error caused by an agent missing a reference.

### B.8 Human validation and iteration

#### B.8.1 Mandatory validation points

| Step | Artifact to validate | Validation question |
|------|---------------------|---------------------|
| B.1 | Topology | "Are all components and couplings represented?" |
| B.2 | Modules | "Is each module implementable in isolation?" |
| B.3 | DAG | "Are the dependencies and parallelizability correct?" |
| B.4 | Classification | "Is the orchestration mode appropriate for each module?" |
| B.5 | Density | "Are the context estimates realistic?" |
| B.6 | Assertions | "Do the interface contracts capture the critical invariants?" |
| B.7 | ISD | "Is each section self-sufficient?" |

#### B.8.2 Part B completeness criterion

The ISD is considered complete when the **naive agent test** is satisfied:

> A Claude Code agent, having only the global file and its module file, can implement the module without asking clarification questions and producing code that passes all interface assertions.

This test can be empirically verified by asking a subagent to implement a non-critical module (e.g., a simple benchmark) from the ISD alone, then verifying the quality of the result.

### B.9 Part B Deliverables — Checklist

- [ ] Topology validated (B.1)
- [ ] Modular decomposition validated (B.2)
- [ ] Dependency DAG validated (B.3)
- [ ] Sequential/parallel classification validated (B.4)
- [ ] Realistic density estimation (B.5)
- [ ] Complete inter-module assertions (B.6)
- [ ] Structured and self-sufficient ISD (B.7)
- [ ] Naive agent test passed on at least one module (B.8)

---

## Part C — Autonomous agents: orchestrated execution

This part — the most voluminous — specifies how agents consume the ISD to produce code. It covers the orchestration architecture, communication protocols, operational templates, and error detection and correction mechanisms.

### C.1 Orchestration architecture

#### C.1.1 Roles

The architecture relies on four distinct roles:

**Lead (human + Claude Code session).** Responsible for: executing the phase plan (B.4), supervising the task list, injecting corrections when a subagent or teammate deviates, validating inter-phase outputs, and making design decisions not anticipated by the ISD.

**Orchestrator (lead Claude Code session).** The main session that coordinates execution. The orchestrator:
- Creates agent teams for parallel phases.
- Assigns subagents for sequential phases.
- Verifies interface assertions after each module.
- Decides to proceed to the next phase (all tasks in the current phase are complete and validated).
- Escalates to the human in case of blockage or ambiguity.

**Builder (subagent or teammate).** Implements a code module from its ISD section. The builder does not make design decisions — it translates the specification into code. If the ISD is ambiguous, the builder flags the ambiguity rather than silently resolving it.

**Validator (subagent).** Verifies the code produced by a builder. The validator does not modify the code — it produces a compliance report. Two types of validation:
- **Formal validation.** The code passes assertions (B.6), unit tests, and satisfies style constraints.
- **Semantic validation.** The code correctly implements the ISD's mathematical specification. This validation is deeper: it verifies that formulas are correctly translated, edge cases are handled, and conventions are respected.

#### C.1.2 Communication topology

```
Human ←→ Orchestrator (lead session)
               ↓ spawn              ↓ spawn
         [Agent Team]          [Subagents]
         ┌──────────┐         ┌──────────┐
         │ Teammate1 │←→│ Teammate2 │    │ Builder  │→│ Validator │
         │ (MOD-001) │  │ (MOD-002) │    │ (MOD-003)│  │ (MOD-003) │
         └──────────┘  └──────────┘    └──────────┘  └──────────┘
              ↕ direct messages               ↑ report
              via inbox                       │
                                     Orchestrator
```

Teammates communicate with each other via the inbox system. Subagents report exclusively to the orchestrator. The orchestrator is the sole point of contact with the human for escalations.

### C.2 Project CLAUDE.md configuration

The `CLAUDE.md` is the context file that all Claude Code instances (orchestrator, teammates, subagents) read at startup. It must be concise (< 5K tokens) and contain only information necessary for **all** agents regardless of their role.

#### C.2.1 Recommended structure

```markdown
# [Project Name]

## Context
[2-3 sentences describing the project and its objective]

## Architecture
[ASCII diagram of the pipeline]

## Critical conventions
- [Convention 1: e.g., "All returns are in log, not arithmetic"]
- [Convention 2: e.g., "Dimension indices start at 0"]
- [Convention 3: ...]

## Invariants — DO NOT VIOLATE
- [INV-001: short description]
- [INV-002: ...]

## Code structure
- `src/module_xxx/`: [description]
- `tests/`: [test naming convention]

## Workflow
- Read `docs/isd/00_global.md` and your module's ISD section BEFORE coding
- Implement interface assertions BEFORE business logic
- Commit after each completed sub-task
- Do not modify files from other modules

## Dependencies
- Python 3.11+
- [List of libraries with versions]

## Tests
- `pytest tests/unit/` for unit tests
- `pytest tests/integration/` for integration tests
- A module is complete only if all its tests pass
```

#### C.2.2 What the CLAUDE.md must NOT contain

- Detailed module specifications (these are in the ISD).
- Complete mathematical formulas (they are in the ISD sections).
- History of design decisions (it is in the DVT).
- Orchestration instructions (they are in the agent prompts).

The CLAUDE.md is an **index** and a **guardrail**, not a specification document.

### C.3 Structure of an ISD section consumable by an agent

Each ISD section (one file per module) follows a rigid template. The rigidity is intentional: it ensures that the agent finds expected information at the expected location, without structural ambiguity.

#### C.3.1 Template

```markdown
# Module [MOD-XXX] — [Module Name]

## Metadata
- **Execution phase:** [1/2/3/4]
- **Orchestration mode:** [lead_session / subagent / teammate]
- **Dependencies:** [MOD-YYY (type D), MOD-ZZZ (type I)]
- **Estimated context density:** [low/medium/high/very_high]
- **Files to produce:** [`src/module_xxx/main.py`, `src/module_xxx/utils.py`, ...]
- **Test files:** [`tests/unit/test_module_xxx.py`, ...]

## Objective
[1-2 paragraphs describing what the module does, why it exists,
and what role it plays in the global pipeline. Sufficient for an agent
to understand the purpose without reading other documents.]

## Inputs
[For each input:]
| Name | Type | Shape | Convention | Source |
|------|------|-------|------------|--------|
| `windows` | `torch.Tensor` | `(N, T, F)` | float32, z-scored per window | MOD-001 output |

## Outputs
[Same format as inputs]

## Technical specifications

### Sub-task 1: [Name]
[Complete description including:]
- Mathematical formula (with definition of all variables)
- Pseudo-code or function signature
- Edge cases and their handling
- Parameters (default value, valid range)

### Sub-task 2: [Name]
[...]

## Applicable invariants
[List of invariants (from A.3) that concern this module,
with the complete description — not a reference]

- **INV-001:** [Complete description of the invariant, including
  the violation consequence and detection method]

## Interface assertions
[Python code for assertions that the module must satisfy]
```python
# assertion_mod_xxx.py
def verify_module_xxx_output(output):
    assert output.shape == (n, K), f"..."
    assert (output >= lower_bound).all(), f"..."
```

## Known pitfalls
[Explicit list of probable errors that an agent may commit]
- "DO NOT use raw MSE without the factor D — this causes posterior collapse"
- "The vol rescaling must be done PER-DATE for historical estimation,
   but at the CURRENT DATE for portfolio construction"

## Required tests
[List of tests that the builder must implement and pass]
1. `test_output_shape`: verifies output shapes
2. `test_edge_cases`: verifies edge case handling
3. `test_mathematical_identity`: verifies a known identity
4. `test_no_lookahead`: verifies absence of look-ahead (if applicable)

## Completion criteria
- [ ] All listed files are created
- [ ] All interface assertions pass
- [ ] All required tests pass
- [ ] Code is documented (docstrings)
- [ ] No unresolved TODOs in the code
```

#### C.3.2 Writing rules

**Self-sufficient formulas.** Each formula redefines its variables locally. Example:

```
Bad: "Apply the formula from Section 4.4 of the DVT"

Good: "The complete loss is:
  L = (D / (2σ²)) · L_recon_weighted + (D/2) · ln(σ²) + L_KL + λ_co(t) · L_co
  where:
  - D = T × F (number of elements per window, T=504 by default, F=2)
  - σ² = exp(log_sigma_sq), learned scalar, clamped to [1e-4, 10], initialized to 1.0
  - L_recon_weighted = (1/|B|) Σ_w γ_eff(w) · MSE(w)
  - γ_eff(w) = 1 + f_c(w) · (γ - 1), with f_c ∈ [0,1] the fraction of days in crisis
  - L_KL = (1/N) Σ_i (1/2) Σ_k (μ²_ik + σ²_ik - ln(σ²_ik) - 1)
  - λ_co(t) follows the curriculum: Phase 1 (λ_max), Phase 2 (linear decay → 0), Phase 3 (0)"
```

**Pitfalls formulated as prohibitions.** Known pitfalls must be formulated as explicit prohibitions, not vague warnings.

```
Bad: "Watch out for the normalization"
Good: "DO NOT normalize by 1/(2σ²) without the factor D. The correct normalization is D/(2σ²)."
```

**Pseudo-code with typed signatures.** Function signatures must specify types and shapes.

```python
def compute_loss(
    x: torch.Tensor,           # (batch, T, F) — input windows, z-scored
    x_hat: torch.Tensor,       # (batch, T, F) — reconstructed windows
    mu: torch.Tensor,          # (batch, K) — encoder mean
    log_var: torch.Tensor,     # (batch, K) — encoder log-variance
    log_sigma_sq: torch.Tensor,# scalar — learned observation noise (log-space)
    gamma_eff: torch.Tensor,   # (batch,) — per-window crisis weight
    lambda_co: float,          # current co-movement weight (curriculum)
    co_mov_loss: torch.Tensor, # scalar — co-movement loss (if applicable)
) -> torch.Tensor:             # scalar — total loss
```

### C.4 Builder-validator protocol

#### C.4.1 Principle

The builder-validator pattern separates code production from its verification. An agent that has implemented a module has developed anchoring biases on its own code — it is less likely to detect its own errors than a fresh agent.

#### C.4.2 Sequence

```
1. Orchestrator assigns the module to the Builder (subagent)
   → Prompt: complete ISD section + global file

2. Builder implements the code
   → Produces: source files + unit tests
   → Intermediate commit

3. Orchestrator assigns the code to the Validator (separate subagent)
   → Prompt: ISD section + Builder's code + interface assertions
   → The Validator does not modify the code

4. Validator produces a report:
   a. PASS — all assertions satisfied, code conforms to the ISD
   b. FAIL — list of non-conformities with ISD references

5a. If PASS → Orchestrator marks the module as complete
5b. If FAIL → Orchestrator sends the report back to the Builder for correction
   → Return to step 2 (max 3 iterations, then human escalation)
```

#### C.4.3 Builder prompt

```markdown
## Role
You are a Builder. Your task is to implement module [MOD-XXX] by strictly
following the ISD specification below. You make no design decisions —
you translate the specification into code.

## Rules
1. Read the ISD section IN ITS ENTIRETY before writing a single line of code.
2. Implement interface assertions FIRST — they define the contract.
3. Implement sub-tasks in the order specified by the ISD.
4. For each sub-task, write the test BEFORE the implementation (TDD).
5. Commit after each completed sub-task (message: "MOD-XXX: sub-task N — [description]").
6. If the ISD is ambiguous on a point, DO NOT guess — flag the ambiguity
   in a comment `# AMBIGUITY: [description]` and continue with
   the most conservative interpretation.
7. Do not prematurely optimize. Favor clarity and ISD conformity.
8. Verify each implemented formula against the ISD formula — character by character.

## ISD Specification
[Complete ISD section for the module]

## Global Context
[Contents of docs/isd/00_global.md]
```

#### C.4.4 Validator prompt

```markdown
## Role
You are a Validator. Your task is to verify that the code of module [MOD-XXX]
conforms to the ISD specification. You do NOT modify the code —
you produce a compliance report.

## Verification procedure
1. For each ISD sub-task, verify that:
   a. The code implements the exact specified formula (verify term by term).
   b. Types and shapes match the specification.
   c. Edge cases are handled as specified.
   d. Parameters have the specified default values and ranges.

2. For each applicable invariant, verify that:
   a. The invariant is respected in the code.
   b. An assertion explicitly verifies it.

3. For each listed pitfall, verify that:
   a. The code does NOT fall into the pitfall.

4. Run the tests and verify that they all pass.

5. Check `# AMBIGUITY:` comments and assess whether the
   Builder's chosen interpretation is acceptable.

## Report format
```yaml
module: MOD-XXX
status: PASS | FAIL
issues:
  - severity: CRITICAL | MAJOR | MINOR
    location: "file:line"
    isd_reference: "Sub-task N, formula X"
    description: "..."
    recommendation: "..."
ambiguities_found:
  - location: "file:line"
    builder_interpretation: "..."
    assessment: "acceptable | needs_clarification"
```

## ISD Specification
[Complete ISD section for the module]

## Code to verify
[Code produced by the Builder]
```

### C.5 Execution phase management

#### C.5.1 Per-phase protocol

**Initialization of a parallel phase (Agent Team).**

```
1. Orchestrator creates the agent team:
   "Create an agent team '[phase_name]' with N teammates."

2. For each teammate, the orchestrator provides:
   - The ISD section for the assigned module
   - The global file (00_global.md)
   - Outputs from previous phase modules (if D dependency)
   - Frozen interfaces (if I dependency)

3. Each teammate implements its module autonomously.

4. Synchronization: the orchestrator waits for all teammates
   to finish (TaskList: all items completed).

5. Inter-module validation:
   - Run interface assertions between the phase's modules.
   - If assertions fail, identify the faulty module and relaunch.

6. Shutdown of the agent team.
7. Phase commit: "Phase N complete — modules MOD-XXX, MOD-YYY, MOD-ZZZ"
```

**Execution of a sequential phase (Subagents).**

```
For each module in the phase (in dependency order):
  1. Spawn Builder subagent with prompt C.4.3
  2. Wait for completion
  3. Spawn Validator subagent with prompt C.4.4
  4. If FAIL: send back to Builder (max 3 iterations)
  5. If PASS after iterations: mark the module as complete
  6. Run interface assertions with previous modules
  7. Commit: "MOD-XXX complete — validated"
```

#### C.5.2 Inter-phase transition management

Before transitioning from phase $n$ to phase $n+1$:

1. **Verify completion.** All modules in phase $n$ are marked as complete and validated.
2. **Run inter-module integration tests.** Interface assertions between phase $n$ modules and those from previous phases pass.
3. **Git checkpoint.** Tag: `phase-N-complete`.
4. **Phase report.** The orchestrator produces a summary: completed modules, passed assertions, encountered problems, decisions made.
5. **Human validation (optional but recommended).** The human reviews the report and approves the transition to the next phase.

### C.6 Inter-agent communication conventions

#### C.6.1 Teammate ↔ Teammate communication (Agent Team)

Teammates communicate via the inbox system to:
- **Signal a ready interface.** "Module MOD-001 is implemented. The output interface is `DataPipelineOutput` in `src/data_pipeline/types.py`."
- **Signal a conflict.** "Module MOD-002 expects input of shape $(T, F)$ but MOD-001 produces $(F, T)$. Who adapts?"
- **Share a utility.** "I implemented a `z_score_per_window()` function in `src/utils/normalization.py` — reusable."

Messages must be factual and actionable. No open discussion — teammates are not a brainstorming channel.

#### C.6.2 Builder/Validator → Orchestrator communication (Subagent)

Subagents report to the orchestrator exclusively via:
- **The produced code** (committed in Git).
- **The validation report** (standardized YAML format, C.4.4).
- **Escalation signals** (`# AMBIGUITY:` comments in code or explicit message "ESCALATION: [description of blockage]").

#### C.6.3 Orchestrator → Human communication

The orchestrator escalates to the human in three cases:
1. **Unresolvable ambiguity.** The ISD does not specify a behavior and the Builder cannot choose conservatively.
2. **Persistent validation failure.** The Builder → Validator cycle has reached 3 iterations without convergence.
3. **Inter-module conflict.** Two modules produce incompatible outputs despite interface contracts.

### C.7 Completion criteria and error handling

#### C.7.1 Module completion criteria

A module is complete if and only if:
1. All files listed in the ISD section are created.
2. All unit tests pass.
3. All interface assertions pass.
4. The Validator has issued a PASS report (or MINOR/MAJOR issues have been resolved).
5. No unresolved `# AMBIGUITY:` comments remain.
6. The code is documented (docstrings for each public function).

#### C.7.2 Phase completion criteria

1. All modules in the phase are complete.
2. Inter-module integration tests pass.
3. The Git checkpoint is created.
4. The phase report is produced.

#### C.7.3 Project completion criteria

1. All phases are complete.
2. End-to-end tests pass (the complete pipeline runs).
3. Global acceptance criteria (A.4) are satisfied.
4. The final holdout (if applicable) has been evaluated.

#### C.7.4 Error handling

**Compilation/runtime error.** The Builder corrects. If the Builder cannot correct in 2 attempts, the orchestrator escalates.

**Interface assertion error.** The orchestrator identifies the responsible module (by comparing the ISD interface specification with actual outputs) and routes back to the Builder/Validator of the faulty module.

**Inter-phase integration error.** The orchestrator compares actual interfaces with interface contracts (B.6). If the contract is respected but integration fails, the contract is probably incomplete — escalation to the human to revise the contract.

**Specification drift.** A Builder has implemented something that passes tests but does not match the ISD's intent (detected by the semantic Validator). The orchestrator sends the report back to the Builder with the precise ISD reference.

### C.8 Operational templates

#### C.8.1 Subagent definition template

```yaml
# .claude/agents/builder.md
---
name: builder
description: "Implements a code module from an ISD section.
             Automatically invoked for implementation tasks."
tools: Read, Write, Edit, Bash, Glob, Grep
model: opus
---

You are a Builder specialized in implementing modules from
ISD specifications.

## Protocol
1. Read the provided ISD section in its entirety.
2. Read the global file (docs/isd/00_global.md).
3. Implement interface assertions first.
4. Implement each sub-task in order, with TDD testing.
5. Commit after each sub-task.
6. Flag ambiguities without resolving them.

## Prohibitions
- Do not modify files from other modules.
- Do not make design decisions not specified by the ISD.
- Do not prematurely optimize.
```

```yaml
# .claude/agents/validator.md
---
name: validator
description: "Verifies code compliance of a module against its
             ISD section. Invoked after each module implementation."
tools: Read, Bash, Glob, Grep
model: opus
---

You are a Validator. You verify code compliance against the ISD
without modifying the code.

## Protocol
1. Read the ISD section and the produced code.
2. Verify each sub-task (formula, types, shapes, edge cases).
3. Run the tests.
4. Produce a standardized YAML report.

## Prohibitions
- Do not modify the source code.
- Do not express opinions on design choices (they come from the ISD).
```

#### C.8.2 Orchestrator prompt template for a parallel phase

```markdown
## Phase [N] — [Phase Name]

### Modules to implement in parallel:
- MOD-XXX ([teammate_name_1]): [short description]
- MOD-YYY ([teammate_name_2]): [short description]
- MOD-ZZZ ([teammate_name_3]): [short description]

### Instructions:
Create an agent team "[phase_name]" with 3 teammates.

Assign to each teammate:
- Its ISD section (docs/isd/XX_module_name.md)
- The global file (docs/isd/00_global.md)
- Outputs from previous phases: [list of available files/interfaces]

Each teammate follows the Builder protocol (read ISD → assertions → sub-tasks → tests → commit).

### Synchronization criteria:
- All modules implemented and tested.
- Inter-module interface assertions verified.
- Git tag: `phase-N-complete`.
```

#### C.8.3 Orchestrator prompt template for a sequential phase

```markdown
## Phase [N] — [Phase Name]

### Execution order:
1. MOD-XXX: [short description]
2. MOD-YYY: [short description] — depends on MOD-XXX
3. MOD-ZZZ: [short description] — depends on MOD-YYY

### For each module:
1. Spawn Builder (subagent) with:
   - ISD section: docs/isd/XX_module_name.md
   - Global file: docs/isd/00_global.md
   - Outputs from previous modules in this phase

2. Wait for Builder completion.

3. Spawn Validator (subagent) with:
   - ISD section (same)
   - Code produced by the Builder

4. Evaluate the Validator's report:
   - PASS → proceed to the next module
   - FAIL → send back to Builder (max 3 iterations)
   - 3 failures → ESCALATION to the human

5. Verify interface assertions with previous modules.

6. Commit: "MOD-XXX validated"
```

### C.9 Anti-patterns and diagnostics

#### C.9.1 Common anti-patterns

**The improvising Builder.** Symptom: code that "works" but does not match the ISD (different algorithm, unspecified parameters, added heuristics). Cause: insufficiently detailed ISD or Builder that does not fully read the specification. Remedy: strengthen the "Known pitfalls" section of the ISD; add assertions that verify expected behavior, not just outputs.

**Superficial validation.** Symptom: the Validator issues PASS but bugs persist. Cause: the Validator only checks that tests pass without verifying semantic conformity (the tests themselves may be insufficient). Remedy: require the Validator's report to explicitly list each verified sub-task with the ISD reference.

**Phantom coupling.** Symptom: a module passes its unit tests but integration fails. Cause: a coupling between modules was not identified in B.1 (coupling degree underestimated). Remedy: add the missing invariant, create the interface assertion, relaunch the faulty module.

**Context inflation.** Symptom: a teammate or subagent produces code of decreasing quality toward the end of its module. Cause: the volume of specification + code exceeds the effective capacity of the context window. Remedy: split the module if possible; otherwise, use a lead session with the human for critical parts.

**Convention divergence.** Symptom: two modules produce numerically different results for the same operation (e.g., covariance calculated differently). Cause: convention not specified in the global file. Remedy: add the convention to CLAUDE.md and to interface assertions.

#### C.9.2 Project health diagnostics

| Signal | Meaning | Action |
|--------|---------|--------|
| > 3 Builder-Validator cycles on a module | Ambiguous or incomplete ISD | Revise the ISD section with the human |
| Integration tests fail after complete phase | Insufficient interface contracts | Add inter-module assertions |
| Teammates communicate excessively | Modules too coupled for parallelization | Switch to sequential mode |
| Frequent `# AMBIGUITY:` comments | Incomplete DVT or ISD | DVT/ISD completion session with the human |
| Produced code significantly longer than expected | Builder adding unspecified logic | ISD conformity review |

### C.10 Part C Deliverables — Checklist

- [ ] Orchestration architecture defined (C.1)
- [ ] Project CLAUDE.md written (C.2)
- [ ] ISD sections complete and self-sufficient (C.3)
- [ ] Builder and Validator subagents configured (C.4, C.8.1)
- [ ] Phase prompts prepared (C.5, C.8.2, C.8.3)
- [ ] Communication conventions documented (C.6)
- [ ] Completion criteria formalized at each level (C.7)
- [ ] Anti-pattern list shared with the orchestrator (C.9)
- [ ] Full execution of all phases
- [ ] End-to-end tests passed
- [ ] Global acceptance criteria satisfied

---

## Appendix — Glossary

| Term | Definition |
|------|-----------|
| **DVT** | Technical Vision Document — source document written by the user describing the project |
| **ISD** | Implementation Specification Document — compilation of the DVT into self-sufficient specifications for agents |
| **Module** | Code unit that is independently implementable and testable |
| **Functional component** | Logical unit of the pipeline (may not correspond 1:1 to a module) |
| **Invariant** | System property that must be true at all times |
| **Interface assertion** | Formal verifiable test checking a contract between modules |
| **Builder** | Agent that implements a module from the ISD |
| **Validator** | Agent that verifies code compliance against the ISD |
| **Orchestrator** | Lead session that coordinates execution of phases |
| **Lead session** | Direct Claude Code session, supervised by the human |
| **Subagent** | Focused agent that reports to the lead, without inter-agent communication |
| **Teammate** | Autonomous Claude Code instance in an Agent Team, with inter-agent communication |
| **DAG** | Directed Acyclic Graph |
| **Coupling** | Degree of interdependence between two components (scale 0–4) |
| **Context density** | Volume of specification that an agent must maintain in memory to implement a module |
| **Self-sufficiency** | Property of an ISD section containing all necessary information without external references |
| **Phase** | Group of modules executable (sequentially or in parallel) before synchronization |
| **Checkpoint** | Git tag marking the validated completion of a phase |

---

*General methodology document — applicable to any project satisfying the scope criteria (Introduction). Instantiation on a specific project produces Document 2: the operational ISD.*
