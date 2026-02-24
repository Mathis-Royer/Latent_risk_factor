# Strategic Paths After V2 Diagnostic — Decision Framework

> **Context:** Two diagnostic runs completed. V1 (pre-fix): Sharpe -0.854. V2 (post-8-fixes): Sharpe -1.584 (86% worse). All 6 benchmarks positive (Sharpe 0.28–0.67). Phase 0 revert applied. This document details the three paths forward after the clean Phase 1 diagnostic run.

> **Decision trigger:** The clean run (Phase 1) must complete first. The metrics from Phase 1 determine which path to follow.

---

## Path A — Incremental Improvement

### Entry condition

Phase 1 clean run shows:
- Cross-sectional R² ~ 10% (V1 level recovered)
- Sharpe in range [-0.5, 0.0]
- AU uncapped ~ 50–64
- Top eigenvalue concentration < 70%
- Solver partially converges (grad_norm < 5.0)

### Rationale

If reverting V2 changes restores V1 performance, it proves the VAE's latent space does carry *some* cross-sectional information (CS R² ~ 10%). While 10% is far below production factor models (Barra USE4: 25–40%), it is non-zero and may be improvable through targeted changes that do NOT alter the encoder architecture.

The key insight: the per-stock temporal VAE discovers temporal regimes (volatility clustering, mean-reversion patterns, trend persistence). Stocks sharing similar temporal dynamics DO share some common risk exposure — but the mapping is weak. Strengthening this mapping requires either (a) reducing noise by focusing on a tighter universe where temporal similarity → risk similarity is stronger, or (b) adjusting the information bottleneck so the encoder retains more cross-sectionally useful features.

### Actions (ordered by expected impact)

#### A.1 — Reduce universe size (4,500 → 1,000–2,000)

**What:** Restrict the investment universe to the top 1,000–2,000 stocks by market cap.

**Why this helps:**
- Fewer stocks → higher stock/factor ratio → better conditioned cross-sectional regression
- Large-cap stocks have more liquid markets, lower idiosyncratic noise, and more correlated behavior (sector/macro factors dominate), which means temporal similarity IS a better proxy for cross-sectional risk exposure
- The n=3,303 universe includes many mid/small-cap stocks where firm-specific risk dominates over factor risk. The VAE's temporal fingerprint captures firm-specific patterns that are NOT common factors
- PCA Factor RP already achieves Sharpe 0.511 on this universe — suggesting that *linear* factor structure exists and is exploitable. With fewer stocks, the VAE may capture similar structure

**Expected impact:** CS R² could increase from ~10% to ~15–20% based on the empirical regularity that factor model fit improves with market-cap-weighted universes (Fama-French 1993, Barra USE4 design).

**Risk:** Reduces diversification opportunity. The strategy may become more correlated with traditional factor models (which also focus on large-cap).

#### A.2 — Explore Mode A (beta_fixed=0.1)

**What:** Switch from Mode P (full ELBO with learned sigma²) to Mode A with a fixed KL weight beta=0.1.

**Why this helps:**
- Mode P with beta=1.0 applies full KL regularization pressure, pushing the posterior toward the prior N(0,I). This is optimal for generative modeling but may be suboptimal for factor discovery
- With beta=0.1, the encoder is allowed to deviate more from the prior, potentially encoding richer cross-sectional information in μ
- Literature support: beta-VAE (Higgins et al. 2017) and InfoVAE (Zhao et al. 2019) show that reducing KL pressure can improve disentanglement and information retention in the latent space
- For our use case, we don't need good generation (decoder quality) — we need informative μ vectors that differentiate stocks cross-sectionally

**Expected impact:** More active units (AU may increase), and each unit may carry more unique information. CS R² could improve by 2–5 percentage points.

**Risk:** Too-low beta can cause posterior collapse in the opposite direction (all information in μ, none in z sampling → deterministic autoencoder). Monitor AU carefully.

#### A.3 — Use only n_signal eigenvalues for entropy

**What:** After the signal/noise eigenvalue split (via DGJ optimal shrinkage), restrict entropy optimization to only the n_signal eigenvalues rather than all AU factors.

**Why this helps:**
- Currently, the entropy optimizer tries to equalize risk contributions across ALL retained factors, including noise factors with tiny eigenvalues
- The principle β'_k ∝ 1/√λ_k means noise factors get disproportionately large exposure
- By restricting to n_signal factors, the optimizer only equalizes risk across statistically significant factors
- This directly addresses Root Cause 2 (eigenvalue concentration makes entropy self-defeating) by removing the noise factors from the optimization landscape

**Expected impact:** Solver convergence should improve dramatically (fewer, better-conditioned factors). Portfolio quality depends on whether n_signal factors span enough cross-sectional variance.

**Risk:** If n_signal is very small (e.g., 1–3), entropy optimization becomes trivial or degenerate. Need n_signal ≥ 5 for meaningful diversification.

#### A.4 — Eigenvalue power shrinkage tuning

**What:** Currently EIGENVALUE_POWER = 0.65. Test values in [0.3, 0.5, 0.7, 1.0] systematically.

**Why this helps:** Power shrinkage compresses the eigenvalue spectrum: λ_k → λ_k^p. With p=0.65, the ratio λ₁/λ₂ is compressed from 22.78 to 22.78^0.65 ≈ 8.7 — still highly concentrated. With p=0.3, ratio → 22.78^0.3 ≈ 2.7, which is much more tractable for entropy optimization.

**Risk:** Aggressive compression (p < 0.5) distorts the risk model — the optimizer equalizes contributions to a compressed spectrum that doesn't reflect true risk. The resulting portfolio may have unintended risk exposures.

### Success criteria for Path A

After implementing A.1–A.4:
- CS R² > 15% (minimum for a functional factor model)
- Sharpe > 0.0 (at least non-negative)
- Solver convergence: grad_norm < 0.1 for at least 2/5 starts
- Variance ratio ∈ [0.5, 3.0]

**If NOT met after all actions:** Escalate to Path B.

### Timeline estimate

Each action requires a full diagnostic run (~5–8 hours with training). Total: ~4 runs × 6 hours = 24 hours of compute. Can be partially parallelized if testing independent changes.

---

## Path B — Architectural Change

### Entry condition

Phase 1 clean run shows:
- CS R² < 5% (confirming architectural mismatch)
- Sharpe < -0.5
- OR: Path A exhausted without meeting success criteria

### Rationale

If the per-stock temporal VAE fundamentally cannot produce useful cross-sectional factor exposures, the architecture must change. The evidence from two diagnostic runs and four analysis plans converges on this conclusion:

1. **Per-stock processing eliminates cross-sectional structure by design.** Each stock is encoded independently — the encoder never sees how stocks relate to each other. Cross-sectional factor structure (stocks co-moving because of shared sector/macro exposures) must be inferred implicitly from temporal similarity. This is an extremely weak assumption.

2. **Marchenko-Pastur analysis shows only 1 eigenvalue above the random matrix noise edge.** This means the VAE's exposure matrix B has the same eigenvalue structure as a random matrix plus one signal factor — strong evidence of minimal cross-sectional information content.

3. **PCA Factor RP achieves Sharpe 0.511 on identical data and constraints.** This proves cross-sectional factor structure exists in the data and is exploitable — the VAE just fails to capture it.

### Option B.1 — FactorVAE Architecture (Duan et al. 2022)

**What:** Replace the current per-stock temporal VAE with a cross-sectional FactorVAE that jointly learns factor exposures β and factor returns f.

**How it works:**
- Input: cross-section of stock returns at time t (vector r_t ∈ ℝⁿ)
- Encoder: maps r_t → posterior q(f_t | r_t) over factor returns
- Decoder: reconstructs r_t ≈ B × f_t + ε_t, where B (n×K) is learned globally
- KL regularization: ensures factors are independent and well-identified
- Key difference: B is a parameter of the model, not derived from per-stock μ vectors

**Why this addresses the root cause:**
- The exposure matrix B is learned cross-sectionally — every training step updates B to explain co-movement across ALL stocks simultaneously
- Factor returns f_t are inferred from the cross-section at each time step — this is exactly what factor models do (Fama-MacBeth 1973)
- The VAE's probabilistic framework provides automatic regularization and uncertainty quantification

**Implementation effort:** Major architectural change. Requires:
- New encoder architecture (cross-sectional input instead of per-stock temporal)
- New decoder (B × f + noise)
- Modified training loop (cross-sectional batching by date, not by stock)
- Modified inference (B is directly a model parameter)
- Keep: risk model, portfolio optimizer, benchmarks, diagnostics (downstream unchanged)

**Estimated work:** 2–3 weeks for a senior ML engineer. Core architecture: 1 week. Training integration: 3–5 days. Testing + debugging: 1 week.

**Risk:** FactorVAE is a research paper (2022), not a production system. Replication may surface issues not reported in the paper (convergence, sensitivity to K, scalability to n=3,000+). The paper uses n=50 stocks and K=3 factors — scaling to our setting is untested.

### Option B.2 — Hybrid: PCA Factors + VAE Regime Detection

**What:** Use PCA for the factor model (proven to work: Sharpe 0.511) and the VAE for non-linear regime detection that modulates risk model parameters.

**How it works:**
1. PCA on trailing returns → factor loadings B_PCA, factor covariance Σ_z
2. VAE on temporal windows → regime indicator z_regime (e.g., crisis/calm/transition)
3. Risk model: Σ_assets = B_PCA × Σ_z(regime) × B_PCA' + D_ε(regime)
4. Regime modulates covariance estimation (e.g., shorter lookback in crisis, longer in calm)
5. Portfolio optimization on Σ_assets → w*

**Why this makes sense:**
- Leverages each model's strength: PCA for cross-sectional structure, VAE for temporal non-linearity
- The VAE's temporal fingerprinting is genuinely useful for regime identification — this is what it was already doing well (reconstruction quality was never the issue)
- Risk model modulation is an established technique (DCC-GARCH, regime-switching models)
- Preserves the entire downstream pipeline (portfolio optimizer, benchmarks, diagnostics)

**Implementation effort:** Moderate. Requires:
- Regime extraction module (from VAE latent space → discrete/continuous regime signal)
- Regime-conditional covariance estimation (extend `covariance.py`)
- Keep: VAE training (as-is), PCA factor model (already implemented as benchmark), portfolio optimizer

**Estimated work:** 1–2 weeks. Regime extraction: 3 days. Conditional covariance: 3–5 days. Integration + testing: 3–5 days.

**Risk:** The value-add of regime detection over simple volatility targeting is uncertain. If the VAE's regime signal is redundant with realized volatility (already used in VT), the hybrid approach adds complexity without benefit.

### Option B.3 — Cross-Sectional Training Objective (Minimal Architecture Change)

**What:** Keep the current encoder but add a cross-sectional loss that is computed correctly (unlike the failed V2 attempt).

**How it differs from V2's lambda_cs:**
- V2 computed batch-level R² (512 stocks, K=75 dims → trivially high R²)
- New approach: use a held-out cross-section. At each step, compute μ for ALL stocks in the universe (not just the batch), regress returns on μ, and backpropagate R² loss
- This is expensive (~10× slower training) but actually measures cross-sectional quality

**Why this might work:**
- The encoder architecture is unchanged — we just add a meaningful training signal
- If the encoder CAN learn cross-sectional structure (architecturally possible but not incentivized), this loss will incentivize it

**Why this might fail:**
- The per-stock encoder may lack the capacity to learn cross-sectional relationships — it only sees one stock's history at a time
- Gradient signal from cross-sectional R² must flow through μ to the encoder, which may be too indirect for effective learning

**Implementation effort:** Moderate. Requires:
- Modified training loop to compute full-universe μ at regular intervals
- Correct R² loss on full cross-section
- Careful gradient management (the universe-wide μ computation is expensive)

**Estimated work:** 1 week.

### Recommendation within Path B

| Option | Impact potential | Implementation effort | Risk |
|--------|-----------------|----------------------|------|
| B.1 FactorVAE | High (directly addresses root cause) | High (2–3 weeks) | Medium (scaling uncertainty) |
| B.2 Hybrid PCA+VAE | Medium (pragmatic, leverages both) | Medium (1–2 weeks) | Low (PCA factor model proven) |
| B.3 CS training fix | Low-Medium (may not overcome architecture) | Medium (1 week) | High (may fail for fundamental reasons) |

**Recommended order:** B.3 first (cheapest experiment), then B.2 if B.3 fails (pragmatic + low risk), then B.1 if B.2 insufficient (full solution).

---

## Path C — Adopt PCA Factor RP (Principle 10)

### Entry condition

Any of the following:
- Phase 1 clean run: CS R² < 3% AND Sharpe < -0.5
- Path A exhausted: CS R² < 15% after all incremental improvements
- Strategic decision: compute budget does not justify further VAE exploration

### Rationale

**Principle 10 (from strategy_philosophy.md):** "If the VAE does not outperform PCA factor risk parity after rigorous evaluation, adopt the simpler model. Intellectual honesty demands acknowledging when additional complexity does not add value."

The evidence is compelling:

| Metric | VAE (V1) | VAE (V2) | PCA Factor RP | Verdict |
|--------|----------|----------|---------------|---------|
| Sharpe | -0.854 | -1.584 | **+0.511** | PCA wins by 1.37–2.10 |
| Max drawdown | ~95% | 99.92% | **~30%** | PCA dramatically safer |
| Complexity | 10M params, 5h training | same | Minutes (SVD) | PCA 1000× simpler |
| CS R² | 9.3% | 2.67% | **~25–35%** | PCA much better |
| Solver convergence | 0/1 | 0/5 | **converges** | PCA well-conditioned |

### What this means concretely

1. **Primary strategy becomes PCA Factor RP** — Bai-Ng IC2 for factor count selection, SCA entropy optimization on PCA factors. Already fully implemented as benchmark (`src/benchmarks/pca_factor_rp.py`).

2. **VAE becomes a research tool** — retained for:
   - Non-linear regime detection (crisis early warning)
   - Temporal pattern analysis (academic research)
   - Potential future use in hybrid approach (Path B.2)

3. **Pipeline simplification:**
   - Remove VAE training from production workflow
   - PCA factor extraction: ~30 seconds (vs 5 hours VAE training)
   - Risk model estimation: unchanged (same Ledoit-Wolf, same VT)
   - Portfolio optimization: unchanged (same SCA solver, much better conditioned)

### Implementation

Path C requires minimal code changes — PCA Factor RP is already a fully functional benchmark with identical constraints (INV-012). The implementation work is primarily:

1. **Promote PCA Factor RP from benchmark to primary strategy** — extract from `src/benchmarks/pca_factor_rp.py` into main pipeline, or simply use the benchmark code path as the production strategy.

2. **Update diagnostics** — modify diagnostic report to center on PCA factor model metrics rather than VAE metrics.

3. **Update notebook** — add PCA-centered diagnostic cells, simplify VAE cells to optional/research.

4. **Documentation** — update `project-overview.md` to reflect the architectural decision and its justification.

**Estimated work:** 2–3 days.

### Justification for early invocation

Even before running Phase 1, the case for Path C is strong:

1. **Two independent diagnostic runs with different configurations both produced catastrophically negative Sharpe** — this is not a tuning problem

2. **The theoretical root cause (per-stock temporal encoding ≠ cross-sectional factor model) is well-understood** and cannot be fixed by configuration changes

3. **Opportunity cost:** Each VAE training run takes ~5 hours. Path A requires ~4 runs (24+ hours of compute). Path B requires weeks of development. Meanwhile, PCA Factor RP is ready now.

4. **Research value preserved:** Adopting PCA Factor RP as the primary strategy does not prevent continued VAE research. The VAE code, training pipeline, and diagnostics remain intact.

### When NOT to invoke Path C

- If Phase 1 shows Sharpe > 0.0 (VAE adds some value, even if less than PCA)
- If CS R² > 15% (architecture has potential, worth exploring Path A)
- If the research objective is specifically to validate the VAE approach (academic publication), regardless of performance vs PCA

---

## Decision Matrix

| Phase 1 Result | CS R² | Sharpe | Recommended Path |
|----------------|-------|--------|-----------------|
| V1 recovered, promising | > 10% | > -0.3 | **Path A** (incremental) |
| V1 recovered, still bad | 5–10% | -0.5 to -0.3 | **Path A** then **B** if insufficient |
| Architecture confirmed broken | < 5% | < -0.5 | **Path B** (start with B.3) |
| Overwhelming evidence | < 3% | < -0.5 | **Path C** (Principle 10) |

---

## Appendix: Changes Already Applied (Phase 0)

The following V2 parameters have been reverted to establish a clean baseline:

| Parameter | V2 Value | Reverted To | File | Reason |
|-----------|----------|-------------|------|--------|
| `lambda_cs` | 0.5 | **0.0** | notebook cell 10, config.py | Confirmed harmful — corrupted encoder |
| `lambda_co_max` | 0.5 | **0.1** | notebook cell 10 | Original value, less loss conflict |
| `curriculum_phase1_frac` | 0.40 | **0.30** | notebook cell 29 | More Phase 3 generalization (30/30/40) |
| `au_max_bai_ng_factor` | 1.0 | **0.0** (disabled) | notebook cell 31, config.py | Confirmed harmful — eigenvalue concentration |
| `MOMENTUM_ENABLED` | True | **False** | notebook cell 11, config.py | Principle 1/9 violation + OOS mismatch |
| `MOMENTUM_WEIGHT` | 0.30 | **0.0** | notebook cell 11 | Must be zero when disabled |
| `PHI` | 15.0 | **5.0** | notebook cell 11 | Less concentration penalty = easier optimization |
| `sca_tol` | 1e-8 | **1e-5** | notebook cell 32 | More realistic for ill-conditioned landscape |

**Kept from V2 (beneficial or harmless):**
- VT clamping [0.5, 2.0] — correct anti-cyclical principle
- Entropy gradient clipping — safety net
- Inverse-vol warm start — harmless
- Fast PGD sub-problem solver — performance
- OOS risk model refresh — fairness vs benchmarks
- feature_weights [2.0, 0.5] — neutral
- log_transform_vol — theoretically sound

---

*Document created: 2026-02-24. To be updated after Phase 1 clean diagnostic run with path selection and results.*
