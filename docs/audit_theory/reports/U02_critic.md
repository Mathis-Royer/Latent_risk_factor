I have what I need. The corpus is internally rich on exactly this unit: §2 states three requirements (multiplicity, graph, temporality), while §4.6–4.7, the philosophy and the supporting notes implement a static exposure matrix with an unconditional covariance. Let me write the audit.

# U02 — critic

## Scope read

- `DVT_strategie_facteurs_risque_latents_v4_1.md` — §1 (L1–39), §2 in full (L43–89), §3 (L93–143), §4.2 (L212–303, partial), §4.3 (L293–436), §4.4 (L438–546, partial), §4.6 (L642–676), §4.7 (L678–759), §4.8 metrics/decomposition (L955–965), §5 parameter tables (L1000–1080, partial), §6.1–6.3 (L1080–1127), §6 priority table (L1185).
- `strategy_philosophy.md` — full (L1–185), with emphasis on §3 (L27–40), §4 (L44–56), §6 (L73–82), §7 (L86–132).
- `Benchmark_latent_risk_factor.md` — protocol and metrics (L1–186).
- `ISD_vae_latent_risk_factors.md` — inference/aggregation (L1594–1653), regime metric (L2310), decision matrix (L2697–2705).
- `sigma_z_ewma_justification.md` — full (L1–125).
- `literature_review_strategy.md` — downside risk (L1–70), factor characterization (L200–291).
- `strategy_comparison.md` — Sortino budgets and comparison table (L27–35, L60, L73, L95, L118, L125–153).
- `literature_review_idiosyncratic_risk.md` — two-layer entropy (L239–244, grep).
- `implementation_decisions.md` — stratification decision (L281–293, grep).

## Five questions

**1. Purpose and original intent of this unit (cite the spec).**
The stated purpose is explicit: "Before designing the model, we need to clarify what it must capture… to establish the requirements that the architecture must satisfy" (`DVT_strategie_facteurs_risque_latents_v4_1.md:45`). The unit is therefore a *requirements-derivation* step: from an ontology of risk (multiplicity L47–58, graph topology L60–64, temporality L66–78) derive constraints on the model, then define risk (L80–89). Its downstream consumers are §3 (data choice), §4.3 (latent capacity), §4.6 (aggregation) and §4.7 (risk object). `strategy_philosophy.md:29` restates the same requirement as a "continuous heatmap", and `strategy_philosophy.md:54` states the operative consequence: a full-history composite that "retains" dormant exposures so that "when it reactivates, the portfolio is already diversified against it."

**2. What a state-of-the-art version of this unit would look like for this objective.**
A rigorous "nature of risk" section would not be a list of risk *sources*; it would be a taxonomy of *estimands* and their identification conditions, each with a test. Concretely, it would:
(a) define the risk functional and the estimand — conditional covariance at the rebalancing date versus unconditional covariance — and justify the choice; the distinction between conditional and unconditional dependence is the core of the modern treatment (Ang & Timmermann, 2012, "Regime Changes and Financial Markets", *Annual Review of Financial Economics* 4:313–337, documents regime switching in means, volatilities and correlations) [unverified];
(b) treat tail dependence as a separate object, not as a by-product of correlation, because elliptical/Gaussian dependence has zero tail dependence for any correlation below one (Embrechts, McNeil & Straumann, 2002, "Correlation and Dependence in Risk Management: Properties and Pitfalls", in *Risk Management: Value at Risk and Beyond*, CUP) [unverified];
(c) formalize "graph topology" as conditional independence, i.e. sparsity of the precision matrix, which is estimable and testable (Friedman, Hastie & Tibshirani, 2008, "Sparse Inverse Covariance Estimation with the Graphical Lasso", *Biostatistics* 9(3):432–441) [unverified];
(d) make the loading map state-dependent, which is precisely what deep conditional factor models do: a network maps characteristics/state to loadings, so the loadings are not constant over time (Chen, Pelger & Zhu, 2024, "Deep Learning in Asset Pricing", *Management Science* 70(2):714–750) [unverified]; (Gu, Kelly & Xiu, 2021, "Autoencoder Asset Pricing Models", *Journal of Econometrics* 222(1):429–450) [unverified];
(e) select factors by a risk-relevance criterion with a formal test for weak factors (Onatski, 2010, "Determining the Number of Factors from Empirical Distribution of Eigenvalues", *Review of Economics and Statistics* 92(4):1004–1016) [unverified];
(f) specify how each requirement is falsified, using consistent scoring rules for covariance forecasts (Patton & Sheppard, 2009, "Evaluating Volatility and Correlation Forecasts", *Handbook of Financial Time Series*) [unverified].
By that standard, §2 is a pre-formal sketch: it names the right phenomena but converts none of them into an estimand, and therefore into a test.

**3. What prevents the current unit from reaching it (evidenced, not assumed).**
Four blockages, all evidenced below. First, §2.4 collapses the entire ontology into one sentence — "what matters is the effective correlation of assets, regardless of its origin" (`DVT_strategie_facteurs_risque_latents_v4_1.md:89`) — after which multiplicity, graph and temporality have no operational content. Second, the model class is Gaussian and linear in the factors (`:458`, `:710`), so the two phenomena §2 insists on (tails, episodic activation) are outside the representable set. Third, time enters the risk object through exactly one scalar per stock — the volatility ratio `:728` — while `B` (`:676`) and `Σ_z` (`:752`) are held fixed; factor-level temporal structure is therefore structurally absent. Fourth, no metric in the validation protocol tests any §2 requirement: Layer 1 tests reconstruction and AU, Layer 2 tests explanatory power and covariance error, Layer 3 tests portfolio outcomes (`DVT_strategie_facteurs_risque_latents_v4_1.md:955–965`; `Benchmark_latent_risk_factor.md:162–175`). The requirements are therefore unfalsifiable in the precise sense that no observation could refute them.

**Steelman, then rebuttal.** The strongest defence of the current design is: (S1) the two-feature input gives the encoder access to volatility co-movement that a return-only PCA cannot see; (S2) a nonlinear encoder can represent state-dependent co-movement that constant linear loadings cannot; (S3) KL auto-pruning gives a data-driven factor count, avoiding a human choice of K; (S4) full-history covariance is the correct choice for a long-horizon diversification mandate that refuses to sell after shocks. Rebuttals: (S1) is already controlled for — the benchmark includes "PCA factor risk parity + realized volatility feature" precisely to isolate the feature effect from the model effect (`Benchmark_latent_risk_factor.md:148–158`), and the corpus calls PCA "the most important" benchmark (`:133`). (S2) is the one genuine deep-learning value proposition, but the risk model then *linearizes* the encoder into `Σ = B Σ_z B' + D_ε` (`DVT_strategie_facteurs_risque_latents_v4_1.md:710`), so the only nonlinear object is `B`; and no test isolates regime-conditional performance. (S3) is false in practice by the corpus's own admission: the statistical guard, not the KL, is "the binding constraint" (`:436`). (S4) is contradicted inside the same document (`:1123`). The design has one defensible edge — a learned, possibly state-dependent loading map — and §2 never states it as a requirement.

**4. Which constraints are real and which are self-imposed design choices.**

| Constraint | Status | Evidence |
|---|---|---|
| Nonstationarity of dependence structure | **Real** | `strategy_philosophy.md:11`, `DVT_strategie_facteurs_risque_latents_v4_1.md:1123` |
| Estimation error in a ~85-factor covariance from ~7.3k dates | **Real** | `DVT_strategie_facteurs_risque_latents_v4_1.md:422–436` |
| Price-only, daily, ~30y, ~1000 names, non-CRSP vendor | **Real given the user's data constraint** | `DVT_strategie_facteurs_risque_latents_v4_1.md:212`, `Benchmark_latent_risk_factor.md:13–19` |
| A vulnerability that has not yet produced price co-movement is unobservable in price data | **Real and unavoidable** | `DVT_strategie_facteurs_risque_latents_v4_1.md:117` vs `:136` |
| Variance as the risk functional | **Self-imposed** (definitional fiat at `:89`) | `DVT_strategie_facteurs_risque_latents_v4_1.md:84–89` |
| Gaussian observation + linear factor covariance | **Self-imposed** for tractability | `DVT_strategie_facteurs_risque_latents_v4_1.md:458`, `:710` |
| Mean aggregation over full history | **Self-imposed**, one of many memory mechanisms | `DVT_strategie_facteurs_risque_latents_v4_1.md:656`; alternative already written at `:1103–1109` |
| Static `B` and static `Σ_z` between retrainings | **Self-imposed** | `DVT_strategie_facteurs_risque_latents_v4_1.md:676`, `:752` |
| The structural/style/episodic taxonomy | **Self-imposed ontology imported from descriptor models** | `DVT_strategie_facteurs_risque_latents_v4_1.md:70–74` |
| K = 200 ceiling and AU guard ≈ 85 | **Self-imposed capacity choice** | `DVT_strategie_facteurs_risque_latents_v4_1.md:420–436` |
| Fundamental data excluded from discovery | **Self-imposed**, argued not demonstrated | `DVT_strategie_facteurs_risque_latents_v4_1.md:130` |
| The tree-versus-graph framing | **Self-imposed false dichotomy** | `DVT_strategie_facteurs_risque_latents_v4_1.md:62–64` |

**5. What should be removed, merged or moved (not added).**
(i) **Remove** §2.1's "infinity of risk sources" table as a requirements source (`:49–58`). It is a taxonomy of labels, not a constraint; it is contradicted by the finite capacity and the binding 85-factor guard. Replace it with one explicit assumption: the return covariance is low-rank plus sparse plus state-dependent, and state the test of each property.
(ii) **Remove** §2.4's "effective correlation, regardless of origin" as the *sole* risk definition (`:89`), or, if kept, **remove** every downstream claim about tails, drawdowns and dormant protection — because those are not representable in the object that sentence defines. The corpus must choose one risk functional; today it uses variance to build and drawdown to justify (`:74`, `Benchmark_latent_risk_factor.md:174–175`).
(iii) **Merge** §2.2 (graph) and §2.3 (temporality) into a single requirement: *the dependence structure is a time-indexed conditional graph*. These are two descriptions of the same missing object, and splitting them lets each be asserted without being tested. This merge also converts the unit from description into an estimand with a test (conditional independence edges; regime-conditional covariance).
(iv) **Move** the "protection before the damage" claim (`:78`; `strategy_philosophy.md:52–54`) out of the requirements section into a hypothesis section, with the admission that it can only cover episodes already present in the 30-year sample, and with a sign and calibration test.
(v) **Remove** mean aggregation as the default (`:656`). If the episodic claim is retained, an unweighted mean over a ~15-effective-sample history is the one aggregation that guarantees dilution; a max/mixture over regimes is the structurally consistent alternative, and the corpus already contains the machinery for an EWMA variant (`:1103–1109`) without drawing the conclusion.
(vi) **Remove** the descriptor-based evidence (`:70–72`) or replace it with evidence about price-derived latent factors; as written it imports the ontology and the persistence of the paradigm the strategy claims to reject (`:37`).
(vii) **Move** the MDD and crisis-return metrics from the ex-post score into the risk model or the objective (`DVT_strategie_facteurs_risque_latents_v4_1.md:961`, `Benchmark_latent_risk_factor.md:175`), or drop the drawdown rationale.
(viii) **Remove** the three-way temporal taxonomy as an architectural premise (`:68–74`); the corpus already proposes the correct replacement — a continuous per-factor persistence estimate (`literature_review_strategy.md:261`).

## Load-bearing assumptions

| ID | Assumption | Where stated | Evidenced? | Comment |
|---|---|---|---|---|
| A1 | Risk is fully characterized by the covariance of returns, and equalizing variance contributions is risk management | `DVT_strategie_facteurs_risque_latents_v4_1.md:89`, `:686`; `strategy_philosophy.md:88` | no | Stated as a definition, then used as if it were a derivation; contradicted by the drawdown motivation at `:74` |
| A2 | Factor exposures and the factor covariance are stationary over the full history | `DVT_strategie_facteurs_risque_latents_v4_1.md:676`, `:752`; `strategy_philosophy.md:56` | no | Contradicted at `:1123` and `strategy_philosophy.md:11` |
| A3 | An unweighted mean over the full history preserves episodic exposures | `DVT_strategie_facteurs_risque_latents_v4_1.md:656`; `strategy_philosophy.md:54`; `ISD_vae_latent_risk_factors.md:1625–1626` | no | Dilutes them to their share of history; see CRI-06 |
| A4 | Proximity in the latent space means similar risk | `strategy_philosophy.md:40` | no | Directly contradicted by `DVT_strategie_facteurs_risque_latents_v4_1.md:724`: 5% and 50% vol stocks get the same `μ` |
| A5 | Selection by marginal KL / explained variance selects risk-relevant factors | `DVT_strategie_facteurs_risque_latents_v4_1.md:436`, `:464`, `:714` | no | Frequency criterion; admitted at `literature_review_strategy.md:243–244` |
| A6 | A Gaussian linear factor covariance represents tail co-movements and episodic activation | implied at `DVT_strategie_facteurs_risque_latents_v4_1.md:458`, `:710`; claimed at `:11`, `:511` | no | Gaussian tail dependence is zero for correlation below one |
| A7 | Persistence of descriptor-based factors transfers to price-derived latent factors | `DVT_strategie_facteurs_risque_latents_v4_1.md:70–72` | no | Circular: the persistence is a property of the descriptor |
| A8 | The effective number of risk sources is ≤ ~85 at this horizon | `DVT_strategie_facteurs_risque_latents_v4_1.md:424–436` vs `:49` | no | "Infinity" premise and finite guard are never reconciled |
| A9 | A 504-day window contains "several market conditions" | `DVT_strategie_facteurs_risque_latents_v4_1.md:301` | no | Contradicted by the 2–5% stress frequency at `:546` |
| A10 | One can be "diversified against" a dormant factor without forecasting its sign or reactivation | `strategy_philosophy.md:54`, `:116` | no | The objective is invariant to the sign of factor exposure; see CRI-07 |
| A11 | Price data reveal structural vulnerabilities before they materialize | `DVT_strategie_facteurs_risque_latents_v4_1.md:58` vs `:136` | no | A vulnerability with no realized co-movement has zero estimated loading |

## Findings

| ID | Type | Severity | Evidence register | Claim (quoted, file:line) | Assessment | Evidence | Cross-unit |
|----|------|----------|-------------------|---------------------------|------------|----------|------------|
| U02-CRI-01 | Requirements do not discriminate model classes (logical) | critical | doc + math | "Before designing the model, we need to clarify what it must capture… to establish the requirements that the architecture must satisfy" — `DVT_strategie_facteurs_risque_latents_v4_1.md:45` | None of the three requirements selects a deep model. "Multiple simultaneous memberships with varying intensities" (`:64`) is exactly a real-valued n×K loading matrix — the object PCA also produces, which the corpus itself identifies as the decisive benchmark (`Benchmark_latent_risk_factor.md:133`). The graph requirement is not satisfied by the chosen model either (CRI-05). The temporality requirement is implemented as mean aggregation (`:656`), which any model can do. A requirement set that is satisfied by the baseline and unsatisfied by the proposed model cannot justify the proposed model. The only model-discriminating requirement — a *state-dependent* loading map — is never stated in §2. | `:45`, `:64`, `:656`, `Benchmark_latent_risk_factor.md:133` | U01, U05, U06, U17 |
| U02-CRI-02 | Mathematical: entropy on unconditional covariance amplifies exactly the rare factors it claims to hedge | critical | math + doc | "c'_k = (β'_{p,k})² · λ_k ≥ 0" — `DVT_strategie_facteurs_risque_latents_v4_1.md:686`; "Empirical covariance of {ẑ_t} over **full history**" — `:752` | Let factor k be dormant with prob. 1−p_k (variance ≈0) and active with prob. p_k (variance λ_a). The eigenvalue used by the optimizer is the *unconditional* variance λ_k ≈ p_k λ_a. Equalizing c_k = β_k²λ_k = C/AU across factors gives β_k ∝ 1/√(p_kλ_a). Conditional on activation the contribution is c_k^act = β_k²λ_a = (C/AU)/p_k. If only k is active and others stay at their unconditional levels, its share of conditional risk is 1/(AU·p_k + 1 − p_k). With AU=80 and p_k=0.05: **20.2%** of conditional total risk from one factor, while the reported H(w) — computed on the unconditional Σ — shows 1/80 = 1.25% and therefore reports near-perfect diversification. The "bet big on weak factors" principle (`strategy_philosophy.md:116`) is precisely this amplification. The diversification metric is measuring the average, the risk is realized in the tail. **DL value and test:** a state-conditional deep factor model can estimate Σ_z(s_t) and p_k; a constant-loading linear model cannot. Discriminating experiment: on crisis days only, regress realized factor contribution shares on λ-decile; the unconditional solution predicts the lowest-λ quintile's share exceeds its target by ≈1/p; a calibrated conditional model predicts no such excess, plus a reliability test (Brier score) on factor-activation events. | `:686`, `:752`, `strategy_philosophy.md:116`, `strategy_philosophy.md:122` | U06, U11, U12, U17 |
| U02-CRI-03 | Internal contradiction: tail risk claimed but Gaussian model | critical | math + literature | "A crisis weighting term in the loss function prioritizes the capture of tail-risk patterns" — `DVT_strategie_facteurs_risque_latents_v4_1.md:11`; "what matters is the effective correlation of assets, regardless of its origin" — `:89` | The risk object consumed by the optimizer is linear-Gaussian (`:458`, `:710`). For a bivariate normal with correlation ρ<1, the tail-dependence coefficient is λ_U = lim_{u→1} P(Y>F_Y^{-1}(u) | X>F_X^{-1}(u)) = 0 (Embrechts, McNeil & Straumann 2002 [unverified]). A Gaussian factor model can therefore represent *correlation that rises with volatility*, but not tail dependence or asymmetric stress co-movement, which §2 lists as a distinct requirement (`:87`) and which §4.4 claims "emerge" in Phase 3 (`:511`). Nothing in the Phase-3 objective references tails: the only dependence term is set to zero (`:508`). The corpus itself concedes the gap — "Crisis weighting… partially captures this, but not explicitly measured" (`literature_review_strategy.md:219`). Either tails matter, in which case the risk functional is wrong, or they do not, in which case the crisis-weighting narrative and the episodic justification must be withdrawn. **DL value and test:** a deep quantile or copula-VAE risk head can represent joint exceedances a Gaussian factor model cannot; test by out-of-sample joint-exceedance likelihood and tail-coverage of predicted CVaR. | `:11`, `:87`, `:89`, `:458`, `:508`, `:511`, `:710`, `literature_review_strategy.md:219` | U07, U11, U12 |
| U02-CRI-04 | Mathematical: no factor-level temporal structure exists | critical | math + doc | "Between retrainings, B is **fixed** — factor dimensions have consistent meaning" — `DVT_strategie_facteurs_risque_latents_v4_1.md:676`; "Empirical covariance… over **full history**" — `:752` | The model's date-t covariance is Σ_t = D_t B Σ_z B' D_t + D_ε with D_t = diag(σ_{i,t}/σ̄_t) (`:728`, `:757`) and Σ_z, D_ε date-invariant. Hence the *factor* correlation matrix corr(Σ_z) is constant, and asset correlations vary only as ρ_ij,t = R_i R_j C_ij / √((R_i²C_ii+D_ii)(R_j²C_jj+D_jj)). Two consequences. (a) A specific episodic factor cannot activate: its direction and covariance are frozen at their full-history average. (b) When all R_i rise in a common vol spike, ρ_ij → C_ij/√(C_iiC_jj) monotonically — the model *cannot fail* to exhibit "crisis contagion", because the increase is a functional-form artifact, not an estimated feature. This makes the central empirical claim of §2.3 unfalsifiable in the worst way: the model will always look as if it captures episodic correlation. D_ε is a full-sample average (`:753`), so crisis idiosyncratic variance is understated. **DL value and test:** a regime-conditional/mixture-of-experts loading and covariance (Σ_z(s_t), B(s_t)) is representable by a deep conditional model and not by constant PCA; test by regime-conditional covariance loss and by whether the estimated state variable improves crisis-day covariance forecasts over the mechanical R_t rescaling. | `:676`, `:728`, `:752`, `:753`, `:757`, `strategy_philosophy.md:50` | U10, U11, U14 |
| U02-CRI-05 | Logical + mathematical: "risk as a graph" is a non sequitur, and the model implies the opposite topology | major | doc + math | "The true structure is closer to a **graph**… which is what a continuous latent space provides" — `DVT_strategie_facteurs_risque_latents_v4_1.md:64`; "This matrix realizes the graph structure from Section 2.2" — `:672` | Two defects. (a) The argument is a false dichotomy: a tree forbids multiple membership only under *hard* assignment; a hierarchical factor model with soft membership is a tree with overlapping leaves, and hierarchical risk parity is exactly such a model. The real requirement is soft assignment, a property of any real-valued loading matrix — not of a latent space. (b) If "graph" means conditional independence, the chosen model implies the *complete* graph. With Σ = BΣ_zB' + D, Woodbury gives Σ^{-1} = D^{-1} − D^{-1}B(Σ_z^{-1}+B'D^{-1}B)^{-1}B'D^{-1}; the second term is dense for dense B, so every pair is conditionally dependent given all others. A sparse supply-chain/contagion topology is the one structure the model cannot represent. | `:62`, `:64`, `:672`, `:710`, `:757` | U05, U06, U11 |
| U02-CRI-06 | Mathematical: mean aggregation dilutes the episodic exposure it is meant to preserve | major | math + doc | "A stock that was exposed to interbank contagion in 2008 retains that exposure in its composite profile… When it reactivates, the portfolio is already diversified against it" — `strategy_philosophy.md:54` | N_eff ≈ T_hist/T ≈ 15 independent windows (`DVT_strategie_facteurs_risque_latents_v4_1.md:651`), and the default is an unweighted mean (`:656`). A two-year crisis occupies ~1/15 ≈ 6.7% of the composite (7.1% by daily-window count). The retained loading on an episode-specific direction is therefore ≈7% of its in-episode value; the corresponding factor eigenvalue is λ_k ≈ 0.07λ_a, i.e. the factor is classified as "weak" *because it is rare*, not because it is low-risk. CRI-02 then converts that small λ into a large β. Dilution and amplification are two halves of one mechanism. The philosophy's protection claim is quantitatively an order of magnitude weaker than its wording. | `:651`, `:656`, `strategy_philosophy.md:54` | U10, U11, U14 |
| U02-CRI-07 | Mathematical: the objective is sign-blind, so "diversified against" is not expressible | major | math + doc | "The portfolio's exposure to each factor is **inversely proportional to the square root of the factor's variance**" — `strategy_philosophy.md:116` | H(w) depends on ĉ_k = β_k²λ_k / Σ_j β_j²λ_j; the risk penalty is w'Σw. Both are invariant under β_k → −β_k. The objective therefore cannot distinguish a portfolio that is long a dormant risk factor from one that is short it: "hedging" and "doubling down" have identical objective value. Protection requires a directional statement (exposure opposite in sign to the factor's loss contribution), which the objective never encodes. For a long-only portfolio the sign is whatever the learned B happens to produce, so the anti-cyclical protection is an accident of the loading signs, not a designed property. This also explains why the corpus never states which sign constitutes protection. | `:686`, `:702`, `strategy_philosophy.md:112–116` | U12, U09, U17 |
| U02-CRI-08 | Misused source / circular evidence | major | doc + literature | "Barra/MSCI risk models report factor stability coefficients above 0.90 for industry exposures (USE4 methodology)" — `DVT_strategie_facteurs_risque_latents_v4_1.md:70`; "Academic studies show half-lives exceeding 25 months for value exposures" — `:72` | Neither line carries an author, year or result beyond the number. Even granting the numbers (USE4 methodology [unverified]; the 25-month half-life claim [unverified]), they are properties of *descriptor-specified* factors: an industry dummy is stable because the classification changes rarely, and a book-to-price exposure is persistent because it is a slow-moving accounting ratio. Neither says anything about the temporal behaviour of price-derived latent factors. The strategy explicitly rejects a priori descriptors (`:37`, `strategy_philosophy.md:21–23`) and then borrows the ontology and persistence of the descriptor paradigm to justify its architecture. This is circular: the evidence is generated by the construction it is meant to validate. | `:37`, `:70`, `:72`, `strategy_philosophy.md:21–23` | U01, U05, U06 |
| U02-CRI-09 | Internal contradiction: stationarity asserted as a principle and denied in the same corpus | major | doc | "Crisis correlations persist permanently in the model" — `strategy_philosophy.md:56` vs "may overweight correlation structures from distant regimes that are no longer representative" — `DVT_strategie_facteurs_risque_latents_v4_1.md:1123` | The philosophy elevates full-history estimation to a principle; the DVT concedes it may be stale, but demotes the correction to optional Iteration 2 with a diagnostic trigger (`:1185`). `strategy_philosophy.md:11` also states that the dependency structure evolves. Three positions coexist: stationary, slow-moving, and regime-shifting. The baseline picks the stationary one; the alternative is only activated after a diagnostic fires, which is a goal-post move — the theory can never be wrong about persistence, because a failure is reclassified as an iteration. | `strategy_philosophy.md:11`, `strategy_philosophy.md:56`, `DVT_strategie_facteurs_risque_latents_v4_1.md:1123`, `:1185` | U11, U09, U14 |
| U02-CRI-10 | Logical: "infinity of sources" versus a binding 85-factor guard; selection is frequency-based | major | doc + math | "There potentially exists an infinity of risk sources" — `DVT_strategie_facteurs_risque_latents_v4_1.md:49`; "This guard is the **binding constraint** in practice, not K itself" — `:436` | If sources are unbounded, every finite truncation is a misspecification; the theory gives no argument for truncating at the 85 dimensions with the *highest average marginal KL*. Marginal KL is a frequency-weighted average over windows and stocks, so a factor with strong but rare activation is ranked last and truncated first; with AU=80, p=0.05 and an active-time KL of 0.15 nats, the average is 0.0075 nats, below the 0.01 threshold. The operating selection rule is therefore "most frequently informative", which is the opposite of the §2.3 requirement to preserve dormant factors. The corpus admits the mechanism: "Recurrence: factors must explain variance across many windows… Strength: high-variance factors are prioritized" (`literature_review_strategy.md:243–244`). | `:49`, `:424–436`, `:714`, `literature_review_strategy.md:243–244` | U06, U05, U18 |
| U02-CRI-11 | Mathematical: a 504-day window cannot contain "several market conditions" | major | math + doc | "T = 504 days (~2 years): long enough for several market conditions (calm, crisis, recovery)" — `DVT_strategie_facteurs_risque_latents_v4_1.md:301` | The corpus states that stress periods are "~2–5% of historical data" (`:546`). A 504-day window therefore contains an expected 10–25 crisis days, and the majority of windows contain none. The local latent vector is thus a calm-regime shape descriptor with occasional contamination, not a "risk regime" sample. This is load-bearing: the entire local-then-aggregate design assumes each window carries a coherent risk state. The justification at `:301` is arithmetic that the corpus's own frequency estimate refutes. | `:301`, `:546` | U04, U05, U11 |
| U02-CRI-12 | Internal inconsistency: the risk functional used to build is not the one used to justify | major | doc | "what matters is the **effective correlation of assets**" — `DVT_strategie_facteurs_risque_latents_v4_1.md:89` vs "These factors drive the most destructive portfolio drawdowns" — `:74` | Construction uses variance/entropy (`:686`, `:710`). Justification uses drawdowns (`:74`), the headline benchmark target is maximum drawdown and crisis-period return (`Benchmark_latent_risk_factor.md:174–175`), and the comparison document proposes Sortino budgets and semi-variance (`strategy_comparison.md:27–35`, `:118`). MDD appears only as an ex-post penalty in the hyperparameter score (`DVT_strategie_facteurs_risque_latents_v4_1.md:961`), never in the objective. A portfolio can therefore maximize the stated objective while failing the stated purpose; the theory provides no reason why the maximizer of H should have a smaller drawdown. The literature review's defence — symmetric covariance is "defensible for a diversification-focused (mu=0) strategy" (`literature_review_strategy.md:16`) — is a non sequitur: mu=0 removes the return term, it does not make variance the right loss functional. | `:74`, `:89`, `:686`, `:961`, `Benchmark_latent_risk_factor.md:174–175`, `strategy_comparison.md:118`, `literature_review_strategy.md:16` | U12, U16, U17 |
| U02-CRI-13 | Unfalsifiable claim: no metric tests the unit's requirements | major | doc + hypothesis | "**Regime decomposition.** Aggregate metrics separately for crisis and calm folds" — `DVT_strategie_facteurs_risque_latents_v4_1.md:961` | The only crisis-related tests are portfolio outcomes: crisis-period return (`Benchmark_latent_risk_factor.md:175`) and regime MSE (`DVT_strategie_facteurs_risque_latents_v4_1.md:961`). Neither identifies the mechanism: a low-beta or low-volatility portfolio passes the crisis-return test without any episodic factor being represented, and reconstruction MSE in crisis windows tests the decoder, not the risk object. There is no test of (a) graph topology, (b) episodic retention, (c) tail dependence. **Testable prediction:** if the episodic mechanism is real, the ex-ante exposure to a factor recovered from a pre-identified historical episode should predict OOS crisis-day losses in a cross-sectional regression controlling for market beta and volatility; a zero coefficient refutes the mechanism. If the graph claim is real, a sparse precision estimated on the residuals should reject the complete-graph null. | `:961`, `Benchmark_latent_risk_factor.md:175`, `literature_review_strategy.md:219` | U16, U14, U15 |
| U02-CRI-14 | Logical: risk sources that have not yet materialized get zero estimated loading | major | math + doc | "**Structural** — Financial leverage, short-term funding dependency" — `DVT_strategie_facteurs_risque_latents_v4_1.md:58` versus "The baseline uses F = 2 (returns + realized volatility)" — `:303` | §2.1 names leverage and funding dependency as risk dimensions; the input is two price-derived features and the exposure estimate is a function of realized co-movement. If a vulnerability has produced no co-movement in the sample, its estimated latent direction has μ ≈ 0 or λ ≈ 0, so c_k = β_k²λ_k ≈ 0 for every w and the factor is outside the objective. Protection against it is therefore impossible by construction, not merely imperfect. This is acknowledged indirectly: the credit-tightening example is presented as the *case for fundamental data* (`:117`), while §3.4 excludes fundamental data from discovery (`:136`). §2.1's own example is inconsistent with §3.4's resolution. | `:58`, `:117`, `:136`, `:303` | U03, U04, U05 |
| U02-CRI-15 | Mathematical: arithmetic error in the auto-pruning threshold | minor | math | "for D = 504 and σ² ≈ 0.5, requires explaining at least ~0.1% of signal variance per dimension" — `DVT_strategie_facteurs_risque_latents_v4_1.md:464` | The latent dimension survives if its reconstruction gain exceeds its KL cost: D·ΔMSE/(2σ²) > 0.01, so ΔMSE > 0.01·2·0.5/504 ≈ 1.98×10⁻⁵. On per-window normalized data with unit variance, that is 0.002%, not 0.1% — a factor of ~50. To obtain 0.1% one would need σ² ≈ MSE ≈ 0.01, inconsistent with the stated σ² ≈ 0.5. Consequence: the KL threshold is far more permissive than advertised, so AU will be large and the operative selection rule becomes the AU_max_stat truncation (CRI-10) — which reinforces, rather than weakens, the finding that the model does not "decide" the factor count. | `:464`, `:436`, `strategy_philosophy.md:73–82` | U06, U07, U18 |
| U02-CRI-16 | Definitional inconsistency | minor | doc | "**Structural** — Financial leverage, short-term funding dependency" — `DVT_strategie_facteurs_risque_latents_v4_1.md:58` vs "**Structural factors** (industry, geography, supply chain)" — `:70` | The same label denotes two different sets six lines apart; "Temporal" in §2.1 (liquidity vs demographics, `:55`) overlaps with the three-way temporal classification of §2.3. The dimensions are neither mutually exclusive nor exhaustive, so they cannot generate distinct architectural requirements — which is consistent with CRI-01: the taxonomy is rhetorical, not analytical. | `:55`, `:58`, `:68–74` | U01, U05 |

## Root-cause hypotheses

**RC1 — The estimand is never defined (conditional vs unconditional dependence).** §2.4's sentence "the relevant risk is that which materializes in price co-movements" (`DVT_strategie_facteurs_risque_latents_v4_1.md:89`) is compatible with the conditional covariance at the rebalancing date, with the unconditional covariance, and with a mixture. Each component then silently picks a different one: the encoder pools all history, `Σ_z` is unconditional (`:752`), the rescaling is a conditional scalar (`:728`), the objective uses unconditional eigenvalues (`:686`), and the justification uses conditional drawdowns (`:74`). This single ambiguity generates CRI-02, CRI-04, CRI-06, CRI-09, CRI-12 and the crisis-metric confusion in CRI-13. *Structural alternatives:* (a) declare the conditional covariance at the rebalancing date the estimand, and make the model state-conditional (mixture-of-experts over a learned state variable) — gains coherence, costs a new identification problem (state estimation error) and more parameters; (b) declare the unconditional covariance the estimand and delete the episodic, anti-cyclical and tail claims — gains honesty and statistical power, costs the strategy's differentiating narrative; (c) a two-object architecture: unconditional covariance for allocation plus a separate conditional tail model used as a constraint — separates the two questions, costs a second model and a coupling rule.

**RC2 — A frequency/variance criterion is substituted for a risk-relevance criterion.** Marginal KL (`:714`), explained variance (`:464`), truncation by descending average KL (`:436`) and PCA comparison by eigenvalue (`Benchmark_latent_risk_factor.md:137`) all rank factors by how often or how much they move, not by how much they can hurt. This generates CRI-05 (the graph is dense because nothing prefers sparsity), CRI-06, CRI-10 and CRI-15. *Structural alternatives:* (a) keep KL for capacity but add an explicit risk-relevance gate — a factor survives only if it improves out-of-sample tail-covariance or crisis-conditional covariance forecast; gains alignment with the mandate, costs statistical power at 34 folds; (b) replace factor counting with a sparse precision graph selected by conditional-independence tests — gains a testable topology, costs an n×n estimation problem and stability issues; (c) rank factors by their worst-state contribution rather than their average contribution — gains tail alignment, costs sensitivity to the state definition.

**RC3 — Requirements were retrofitted to a pre-chosen architecture.** §2 announces requirements (`:45`) whose only possible empirical content is the covariance, and whose only possible architectural content is a loading matrix — both already decided. Nothing in §2 could have produced a different architecture, and nothing in §4.8 could refute §2. This generates CRI-01, CRI-03, CRI-08 and CRI-13. *Structural alternatives:* (a) invert the order — write the falsifiable tests and the estimand first, then state the model class and its assumptions; gains falsifiability, costs the rhetorical bridge from "nature of risk" to the VAE; (b) delete §2 as a requirements section and relabel it "motivation", removing its claim to constrain the design; (c) replace the three descriptive pillars with one conditional-dependence requirement and one test, accepting that this makes the deep-learning case narrower but defensible.

**RC4 — The construction functional and the justification functional are different.** Variance builds the portfolio; drawdown justifies it, scores it and benchmarks it (CRI-12). *Structural alternatives:* (a) adopt a coherent tail functional (CVaR / semi-covariance, or a deep quantile factor model) as the construction object — gains means-ends coherence, costs estimation precision and convexity (`literature_review_strategy.md:52–55`); (b) keep variance in construction but impose a tail budget as a hard constraint — gains tractability, costs a constrained non-convex problem; (c) keep variance and delete the drawdown justification — gains consistency, costs the anti-cyclical selling point.

**RC5 — "Dormancy" is a placeholder for a mechanism that is never modelled.** The word appears where a probability, a sign and a conditional distribution would be required (`strategy_philosophy.md:50–54`, `DVT_strategie_facteurs_risque_latents_v4_1.md:74–78`). Because no activation probability, sign or conditional law is estimated, the claim cannot fail, and the objective is sign-blind (CRI-07). This generates CRI-06, CRI-13 and CRI-14. *Structural alternatives:* (a) replace "dormant protection" with an explicit two-state factor model estimating p_k and λ_a,k, and test calibration — gains falsifiability, costs a state-labelling rule; (b) treat episodic exposure as a robust constraint over an activation set (worst-case factor realization) — gains a decision-theoretic foundation, costs conservatism; (c) drop the claim and reframe the strategy as a variance-diversification strategy — gains honesty, loses the stated raison d'être.

## Verdict

Section 2 is the weakest link in the corpus because it is the only place where the strategy's *purpose* is defined, and it defines it in a way that cannot constrain anything. Its three pillars either are satisfied by the linear baseline (continuous multi-membership is a loading matrix; CRI-01), or are contradicted by the model (the graph is complete, not sparse, CRI-05; factor structure is frozen, CRI-04; tails are structurally zero, CRI-03), or are diluted to a fraction of their stated strength by the aggregation (CRI-06). The single formal statement that survives — "risk is effective correlation" — reduces the entire ontology to a covariance matrix, which is precisely the object the benchmarks estimate. The deepest defect is not any one equation but a missing estimand: because the theory never says whether it estimates a conditional or an unconditional covariance, it simultaneously claims present-structure identification, permanent crisis memory, and adaptation to regime change, and it can never be shown wrong on any of them (RC1). The constructive core is narrow but real: the only requirement that would distinguish a deep model from PCA is that the loading map and the factor covariance be functions of a learned state — and §2 never states it. As a requirements derivation, this unit is not sound; as motivation, it is vivid and partly right about the phenomena it names.

Score: 3/10

## Machine companion

```json
[
  {
    "id": "U02-CRI-01",
    "unit": "U02",
    "role": "critic",
    "type": "requirements do not discriminate model classes",
    "severity": "critical",
    "evidence_register": "doc + math",
    "claim": "Before designing the model, we need to clarify what it must capture to establish the requirements that the architecture must satisfy.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:45",
    "assessment": "Multiple simultaneous membership with continuous intensities is exactly a real-valued n x K loading matrix, which PCA also produces; the graph requirement is not met by the chosen model either; the temporality requirement is implemented as mean aggregation, available to any model. No requirement in Section 2 selects a deep model, and the only discriminating requirement (a state-dependent loading map) is never stated.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:45",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:64",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:656",
      "Benchmark_latent_risk_factor.md:133"
    ],
    "cross_units": ["U01", "U05", "U06", "U17"],
    "root_cause": "RC3: requirements retrofitted to a pre-chosen architecture",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-02",
    "unit": "U02",
    "role": "critic",
    "type": "mathematical: unconditional entropy amplifies rare factors",
    "severity": "critical",
    "evidence_register": "math + doc",
    "claim": "Each factor risk contribution is c'_k = (beta'_k)^2 lambda_k, and the lambda are estimated over full history.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:686",
    "assessment": "With lambda_k = p_k lambda_active, equalizing c_k = C/AU gives a conditional contribution (C/AU)/p_k; with AU=80 and p=0.05 one factor supplies 20.2 percent of conditional total risk while the reported entropy shows 1.25 percent. The diversification metric measures the average, the risk is realized in the tail.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:686",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:752",
      "strategy_philosophy.md:116",
      "strategy_philosophy.md:122"
    ],
    "cross_units": ["U06", "U11", "U12", "U17"],
    "root_cause": "RC1: undefined estimand (unconditional used for a conditional question)",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-03",
    "unit": "U02",
    "role": "critic",
    "type": "internal contradiction: tail risk claimed, Gaussian model used",
    "severity": "critical",
    "evidence_register": "math + literature + doc",
    "claim": "A crisis weighting term in the loss function prioritizes the capture of tail-risk patterns; but what matters is the effective correlation of assets regardless of origin.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:11",
    "assessment": "The risk object is linear-Gaussian. For a bivariate normal with correlation below one the tail dependence coefficient is zero (Embrechts, McNeil and Straumann 2002, unverified), so tail dependence and asymmetric stress co-movement are outside the model class. The Phase-3 objective removes the only dependence term and contains nothing about tails. Either tails matter and the risk functional is wrong, or they do not and the crisis-weighting narrative must be withdrawn.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:87",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:89",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:458",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:508",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:511",
      "literature_review_strategy.md:219"
    ],
    "cross_units": ["U07", "U11", "U12"],
    "root_cause": "RC4: construction functional differs from justification functional",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-04",
    "unit": "U02",
    "role": "critic",
    "type": "mathematical: no factor-level temporal structure",
    "severity": "critical",
    "evidence_register": "math + doc",
    "claim": "Between retrainings B is fixed, and the factor covariance is the empirical covariance over full history.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:676",
    "assessment": "Sigma_t equals D_t B Sigma_z B' D_t plus constant D_epsilon, so the factor correlation matrix is constant and asset correlations vary only as a deterministic function of the volatility ratio vector. A specific episodic factor cannot activate, and correlation increases in a common volatility spike are a functional-form artifact the model cannot fail to produce. The episodic claim is therefore unfalsifiable.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:676",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:728",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:752",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:753",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:757",
      "strategy_philosophy.md:50"
    ],
    "cross_units": ["U10", "U11", "U14"],
    "root_cause": "RC1: undefined estimand; static object used to answer a conditional question",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-05",
    "unit": "U02",
    "role": "critic",
    "type": "logical and mathematical: graph claim is a non sequitur and is contradicted",
    "severity": "major",
    "evidence_register": "doc + math",
    "claim": "The true structure is closer to a graph, which is what a continuous latent space provides; this matrix realizes the graph structure.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:64",
    "assessment": "The tree-versus-graph dichotomy is false: a tree forbids multiple membership only under hard assignment, and soft-membership hierarchical models exist. If graph means conditional independence, the chosen covariance B Sigma_z B' plus diagonal D has a dense precision matrix by Woodbury, i.e. a complete graph; the sparse topology the metaphor implies is the one structure the model cannot represent.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:62",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:64",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:672",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:710"
    ],
    "cross_units": ["U05", "U06", "U11"],
    "root_cause": "RC2: frequency criterion substituted for a structural criterion",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-06",
    "unit": "U02",
    "role": "critic",
    "type": "mathematical: mean aggregation dilutes episodic exposures",
    "severity": "major",
    "evidence_register": "math + doc",
    "claim": "A stock exposed to interbank contagion in 2008 retains that exposure in its composite profile, and when it reactivates the portfolio is already diversified against it.",
    "claim_location": "strategy_philosophy.md:54",
    "assessment": "With about 15 effective independent windows over 30 years and an unweighted mean, a two-year crisis contributes about 6.7 percent of the composite. The retained loading is an order of magnitude below its in-episode value, and the resulting small eigenvalue is then converted into a large exposure by the entropy objective.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:651",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:656",
      "strategy_philosophy.md:54"
    ],
    "cross_units": ["U10", "U11", "U14"],
    "root_cause": "RC5: dormancy is a placeholder, not a modelled mechanism",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-07",
    "unit": "U02",
    "role": "critic",
    "type": "mathematical: objective is sign-blind",
    "severity": "major",
    "evidence_register": "math + doc",
    "claim": "The portfolio exposure to each factor is inversely proportional to the square root of the factor variance.",
    "claim_location": "strategy_philosophy.md:116",
    "assessment": "Both the entropy and the variance penalty are invariant under beta to minus beta, so the objective cannot distinguish hedging a factor from doubling down on it. Protection requires a directional statement the objective never encodes; for a long-only portfolio the sign is an accident of the learned loadings.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:686",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:702",
      "strategy_philosophy.md:112",
      "strategy_philosophy.md:116"
    ],
    "cross_units": ["U12", "U09", "U17"],
    "root_cause": "RC5: dormancy is a placeholder, not a modelled mechanism",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-08",
    "unit": "U02",
    "role": "critic",
    "type": "misused and circular evidence",
    "severity": "major",
    "evidence_register": "doc + literature",
    "claim": "Barra/MSCI risk models report factor stability coefficients above 0.90 for industry exposures; academic studies show half-lives exceeding 25 months for value exposures.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:70",
    "assessment": "No author, year or result beyond the number is given. Even granting the figures (USE4 methodology unverified; 25-month half-life unverified), they are properties of descriptor-specified factors: an industry dummy is stable because the classification rarely changes, and book-to-price is persistent because it is a slow-moving accounting ratio. The strategy rejects a priori descriptors and then borrows their ontology and persistence to justify its design.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:37",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:70",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:72",
      "strategy_philosophy.md:21"
    ],
    "cross_units": ["U01", "U05", "U06"],
    "root_cause": "RC3: requirements retrofitted to a pre-chosen architecture",
    "confidence": "medium"
  },
  {
    "id": "U02-CRI-09",
    "unit": "U02",
    "role": "critic",
    "type": "internal contradiction: stationarity asserted and denied",
    "severity": "major",
    "evidence_register": "doc",
    "claim": "Crisis correlations persist permanently in the model, whereas full history estimation may overweight correlation structures from distant regimes that are no longer representative.",
    "claim_location": "strategy_philosophy.md:56",
    "assessment": "The philosophy makes persistence a principle; the DVT concedes staleness but demotes the correction to an optional iteration with a diagnostic trigger. Three incompatible positions coexist (stationary, slow-moving, regime-shifting), and a failure of persistence is reclassified as an iteration, so the theory cannot be wrong.",
    "evidence": [
      "strategy_philosophy.md:11",
      "strategy_philosophy.md:56",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:1123",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:1185"
    ],
    "cross_units": ["U11", "U09", "U14"],
    "root_cause": "RC1: undefined estimand",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-10",
    "unit": "U02",
    "role": "critic",
    "type": "logical: infinite sources versus binding factor guard",
    "severity": "major",
    "evidence_register": "doc + math",
    "claim": "There potentially exists an infinity of risk sources; yet the statistical guard is the binding constraint in practice, not K itself.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:49",
    "assessment": "If sources are unbounded every finite truncation is a misspecification, and the operative rule is truncation to 85 dimensions by descending average marginal KL, a frequency criterion. A factor active in 5 percent of windows with an active-time KL of 0.15 nats averages 0.0075 nats and is pruned first. The selection rule is the opposite of the requirement to preserve dormant factors.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:49",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:424",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:436",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:714",
      "literature_review_strategy.md:243"
    ],
    "cross_units": ["U06", "U05", "U18"],
    "root_cause": "RC2: frequency criterion substituted for risk relevance",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-11",
    "unit": "U02",
    "role": "critic",
    "type": "mathematical: window cannot contain several regimes",
    "severity": "major",
    "evidence_register": "math + doc",
    "claim": "T = 504 days is long enough for several market conditions (calm, crisis, recovery).",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:301",
    "assessment": "The corpus states stress periods are 2 to 5 percent of data, so a 504-day window contains an expected 10 to 25 crisis days and most windows contain none. The local latent vector is a calm-regime shape descriptor, not a risk-regime sample, which undermines the local-then-aggregate design.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:301",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:546"
    ],
    "cross_units": ["U04", "U05", "U11"],
    "root_cause": "RC1: undefined estimand",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-12",
    "unit": "U02",
    "role": "critic",
    "type": "internal inconsistency: construction versus justification functional",
    "severity": "major",
    "evidence_register": "doc",
    "claim": "What matters is the effective correlation of assets; yet episodic factors drive the most destructive portfolio drawdowns.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:89",
    "assessment": "Variance and entropy construct the portfolio; drawdowns and crisis returns justify and score it. MDD appears only as an ex-post penalty, never in the objective. The claim that symmetric covariance is defensible for a mu=0 strategy is a non sequitur: mu=0 removes the return term, it does not make variance the right loss functional.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:74",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:89",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:961",
      "Benchmark_latent_risk_factor.md:174",
      "Benchmark_latent_risk_factor.md:175",
      "strategy_comparison.md:118",
      "literature_review_strategy.md:16"
    ],
    "cross_units": ["U12", "U16", "U17"],
    "root_cause": "RC4: construction functional differs from justification functional",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-13",
    "unit": "U02",
    "role": "critic",
    "type": "unfalsifiable claim and missing test",
    "severity": "major",
    "evidence_register": "doc + hypothesis",
    "claim": "Regime decomposition aggregates metrics separately for crisis and calm folds; the crisis-period return is the direct test of anti-cyclical protection.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:961",
    "assessment": "The only crisis-related tests are portfolio outcomes and decoder MSE; neither identifies the episodic mechanism, and a low-beta or low-volatility portfolio passes without any episodic factor being represented. Testable prediction: exposure to a factor recovered from a pre-identified historical episode should predict OOS crisis-day losses controlling for market beta and volatility; a zero coefficient refutes the mechanism. A sparse precision estimated on residuals should reject the complete-graph null if the graph claim holds.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:961",
      "Benchmark_latent_risk_factor.md:175",
      "literature_review_strategy.md:219"
    ],
    "cross_units": ["U16", "U14", "U15"],
    "root_cause": "RC3: requirements retrofitted to a pre-chosen architecture",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-14",
    "unit": "U02",
    "role": "critic",
    "type": "logical: unmanifested vulnerabilities have zero estimated loading",
    "severity": "major",
    "evidence_register": "math + doc",
    "claim": "Structural risk includes financial leverage and short-term funding dependency; the baseline uses two price-derived features and prices only for discovery.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:58",
    "assessment": "If a vulnerability has produced no co-movement in the sample, its latent direction has near-zero mean or variance, so its risk contribution is zero for every weight and it is outside the objective. Protection is impossible by construction, not merely imperfect. The credit-tightening example is presented as the case for fundamental data while the resolution excludes fundamental data from discovery.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:58",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:117",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:136",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:303"
    ],
    "cross_units": ["U03", "U04", "U05"],
    "root_cause": "RC5: dormancy is a placeholder, not a modelled mechanism",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-15",
    "unit": "U02",
    "role": "critic",
    "type": "mathematical: arithmetic error in the pruning threshold",
    "severity": "minor",
    "evidence_register": "math",
    "claim": "For D = 504 and sigma squared about 0.5, a dimension must explain at least about 0.1 percent of signal variance to remain active.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:464",
    "assessment": "The survival condition D times delta MSE over 2 sigma squared above 0.01 gives delta MSE above 1.98e-5, i.e. 0.002 percent of unit normalized variance, about fifty times smaller than stated. Getting 0.1 percent would require sigma squared about 0.01, inconsistent with the stated 0.5. The threshold is therefore far more permissive than advertised and the operative selection is the AU_max_stat truncation.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:464",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:436",
      "strategy_philosophy.md:73"
    ],
    "cross_units": ["U06", "U07", "U18"],
    "root_cause": "RC2: frequency criterion substituted for risk relevance",
    "confidence": "high"
  },
  {
    "id": "U02-CRI-16",
    "unit": "U02",
    "role": "critic",
    "type": "definitional inconsistency",
    "severity": "minor",
    "evidence_register": "doc",
    "claim": "Structural means financial leverage and short-term funding dependency in one table and industry, geography and supply chain in another.",
    "claim_location": "DVT_strategie_facteurs_risque_latents_v4_1.md:70",
    "assessment": "The same label denotes two different sets six lines apart, and the temporal dimension of the first table overlaps the three-way classification of the second. The dimensions are neither mutually exclusive nor exhaustive, so they cannot generate distinct architectural requirements, consistent with the finding that the requirements do not discriminate.",
    "evidence": [
      "DVT_strategie_facteurs_risque_latents_v4_1.md:55",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:58",
      "DVT_strategie_facteurs_risque_latents_v4_1.md:70"
    ],
    "cross_units": ["U01", "U05"],
    "root_cause": "RC3: requirements retrofitted to a pre-chosen architecture",
    "confidence": "high"
  }
]
```

END-OF-REPORT
