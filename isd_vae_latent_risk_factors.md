# ISD — VAE Latent Risk Factor Discovery Strategy

## Implementation Specification Document — Instanciation opérationnelle

**Source DVT :** `strategie_facteurs_risque_latents_v4_1.md` (v4.1)
**Source benchmarks :** `Latent_risk_factor_benchmark.md`
**Méthodologie :** ISD Methodology v1.0

---

## 00 — Contexte global et conventions

### Objectif du projet

Implémenter un pipeline end-to-end de construction de portefeuille basé sur la découverte de facteurs de risque latents par un Variational Autoencoder (VAE). Le pipeline comprend : préparation des données financières, entraînement d'un VAE 1D-CNN, inférence des profils de risque composites, estimation d'un modèle de risque factoriel, optimisation de portefeuille par entropie factorielle, et validation par walk-forward sur 30 ans d'historique. Six benchmarks servent de référence.

### Architecture du pipeline

```
[CRSP Data] → [data_pipeline] → windows (N×T×F) + crisis_labels
                                        ↓
                               [vae_architecture] → VAEModel
                                        ↓
                          [loss_function] → loss computation
                                        ↓
                             [training] → trained encoder
                                        ↓
                            [inference] → B (n×K), composite profiles
                                        ↓
                           [risk_model] → B_A, Σ_z, D_ε, Σ_assets
                                        ↓
                   [portfolio_optimization] → w* (optimal weights)
                                        ↓
                        [walk_forward] → OOS metrics across ~34 folds
                                        ↓
                       [benchmarks ×6] → comparative metrics
                                        ↓
                        [integration] → final report, statistical tests
```

### Conventions critiques — NE PAS VIOLER

| ID | Convention | Modules concernés |
|----|-----------|-------------------|
| CONV-01 | **Rendements en log** : $r_t = \ln(P_t^{\text{adj}} / P_{t-1}^{\text{adj}})$, jamais arithmétiques | Tous |
| CONV-02 | **Z-score per-window** : chaque fenêtre (T, F) est normalisée indépendamment (moyenne 0, std 1 par feature) | data_pipeline, loss_function, inference |
| CONV-03 | **Indices 0-based** partout (Python standard) | Tous |
| CONV-04 | **Tenseurs PyTorch** pour le VAE ; **NumPy/arrays** pour le pipeline downstream (risk_model, optimization) | Tous |
| CONV-05 | **Shape des fenêtres** : (batch, T, F) — le temps est la dimension 1, les features la dimension 2 | data_pipeline, vae_architecture, loss_function, training |
| CONV-06 | **σ² est scalaire** (une seule valeur pour tout le modèle), pas vectoriel par feature ou par dimension | loss_function, training, vae_architecture |
| CONV-07 | **AU** = nombre de dimensions actives = |{k : KL_k > 0.01 nats}|, déterminé une fois par retraining | inference, risk_model, portfolio_optimization |
| CONV-08 | **Rescaling duale** : date-specific pour l'estimation historique, current-date pour la construction de portefeuille | risk_model, portfolio_optimization |
| CONV-09 | **Expanding window** pour le training (tout l'historique), pas rolling | walk_forward, training |
| CONV-10 | **Point-in-time** : aucune donnée future dans aucun calcul ; univers reconstitué à chaque date | data_pipeline, walk_forward |

### Invariants critiques

```yaml
invariants:
  - id: INV-001
    category: mathematical
    description: "Le facteur D = T × F DOIT apparaître dans le numérateur de la reconstruction loss.
                  La loss reconstruction est D/(2σ²) · L_recon, PAS 1/(2σ²) · L_recon."
    modules: [loss_function, training, monitoring]
    violation: "Posterior collapse — toutes les dimensions convergent vers le prior,
               AU → 0, le modèle est inutile"
    detection: "AU < 5 après entraînement complet ; σ² → borne inférieure (1e-4)"
    test: "assert loss_recon_coefficient == (T * F) / (2 * sigma_sq)"

  - id: INV-002
    category: mathematical
    description: "σ² est un scalaire appris (pas un vecteur). Un seul paramètre log_sigma_sq,
                  avec σ² = exp(log_sigma_sq), clampé à [1e-4, 10]."
    modules: [vae_architecture, loss_function]
    violation: "Si σ² vectoriel (per-feature ou per-dimension), le balancement auto-régulé
               recon/KL est rompu — chaque dimension a son propre trade-off"
    test: "assert log_sigma_sq.ndim == 0 or log_sigma_sq.numel() == 1"

  - id: INV-003
    category: inter_module
    description: "La matrice B après filtrage a la shape (n, AU) avec AU ≤ AU_max_stat.
                  AU_max_stat = floor(sqrt(2 * N_obs / r_min)) avec r_min = 2, N_obs = T_hist_jours."
    modules: [inference, risk_model]
    violation: "Σ_z sous-estimée (rapport observations/paramètres < 1), optimisation instable"
    test: |
      assert B_A.shape == (n_stocks, AU)
      assert AU <= AU_max_stat

  - id: INV-004
    category: convention
    description: "Rescaling duale : B_A_estimation[i,t] = (σ_i,t / σ_bar_t) · μ_A_i ;
                  B_A_portfolio[i] = (σ_i,now / σ_bar_now) · μ_A_i.
                  NE PAS utiliser la même rescaling pour les deux."
    modules: [risk_model, portfolio_optimization]
    violation: "Utiliser current-date vol pour l'estimation historique attribue incorrectement
               le risque systématique aux résidus, gonflant D_ε"
    test: "Vérifier que z_hat_t utilise B_A_t (date-specific) et que Σ_assets utilise B_A_port (current)"

  - id: INV-005
    category: safety
    description: "Aucun look-ahead. Les données de test ne sont jamais vues pendant l'entraînement.
                  L'embargo de 21 jours sépare training et OOS. Le VIX threshold est calculé
                  sur expanding window du training set uniquement."
    modules: [data_pipeline, walk_forward, training]
    violation: "Backtest invalide — résultats artificiellement optimistes"
    test: |
      assert all(train_dates < embargo_start)
      assert all(test_dates > embargo_end)
      assert vix_threshold_computed_on <= training_end_date

  - id: INV-006
    category: mathematical
    description: "Les trois modes de loss (P/F/A) sont MUTUELLEMENT EXCLUSIFS.
                  Mode P : σ² appris, β=1 fixe.
                  Mode F : σ²=1 gelé, β_t annealing, D/2 scaling retenu.
                  Mode A : σ² appris, β>1 fixe.
                  NE PAS combiner σ² appris avec β annealing."
    modules: [loss_function, training]
    violation: "Interaction imprévisible σ²/β — le modèle compense le double KL pressure"
    test: "assert not (sigma_sq_learned and beta_annealing_enabled)"

  - id: INV-007
    category: mathematical
    description: "L'entropie H(w) est calculée dans la base des FACTEURS PRINCIPAUX de Σ_z
                  (après rotation V de l'eigendecomposition Σ_z = VΛV^T), PAS dans la base
                  latente brute. Les contributions c'_k = (β'_p,k)² · λ_k sont toujours ≥ 0."
    modules: [portfolio_optimization, risk_model]
    violation: "Contributions négatives → entropie indéfinie → solver diverge"
    test: |
      assert all(c_prime_k >= 0)
      assert all(eigenvalues >= 0)

  - id: INV-008
    category: inter_module
    description: "Le ratio σ_i,t / σ_bar_t est WINSORISÉ cross-sectionnellement à chaque date
                  aux percentiles [P5, P95] AVANT rescaling. Le même ratio winsorisé est utilisé
                  pour B_A_estimation ET B_A_portfolio."
    modules: [risk_model]
    violation: "Un stock à R=15 reçoit 225× le poids de régression, le bruit idiosyncratique
               contamine Σ_z"
    test: "assert all(ratio >= percentile_5) and all(ratio <= percentile_95)"

  - id: INV-009
    category: mathematical
    description: "Le gradient de H par rapport à w est :
                  ∇_w H = -(2/C) · B' · φ, avec φ_k = λ_k · β'_k · (ln(ĉ'_k) + H)
                  où B' = B_A_port_rotated, C = Σ_k λ_k (β'_k)²"
    modules: [portfolio_optimization]
    violation: "Convergence SCA incorrecte, solution sous-optimale"
    test: "Vérifier que ∇H = 0 quand ĉ'_k = 1/AU pour tout k (maximum entropy)"

  - id: INV-010
    category: inter_module
    description: "Le curriculum co-movement a 3 phases : Phase 1 (λ_co = λ_co_max, ~30% epochs),
                  Phase 2 (décroissance linéaire → 0, ~30%), Phase 3 (λ_co = 0, ~40%).
                  Le batching change entre phases : synchrone+stratifié (Ph 1-2), random (Ph 3)."
    modules: [loss_function, training]
    violation: "Batching synchrone en Phase 3 = gradient variance inutilement élevée ;
               batching random en Phase 1 = co-movement loss non-calculable"

  - id: INV-011
    category: mathematical
    description: "La validation ELBO exclut γ (crisis weighting) et λ_co (co-movement loss).
                  Elle inclut σ². Formule : L_val = D/(2σ²)·L_recon^(γ=1) + (D/2)·ln(σ²) + L_KL"
    modules: [training]
    violation: "Selection bias vers les folds riches en crises si γ est inclus dans la validation"

  - id: INV-012
    category: convention
    description: "Les contraintes de portefeuille sont identiques entre le VAE et tous les benchmarks :
                  long-only, fully invested, w_max=5%, w_min=0.10% ou 0, P_conc, P_turn, τ_max=30%"
    modules: [portfolio_optimization, benchmarks]
    violation: "Comparaison invalide — les différences reflètent les contraintes, pas le modèle"
```

### Glossaire des symboles

| Symbole | Définition | Valeur par défaut |
|---------|-----------|-------------------|
| $n$ | Nombre de stocks dans l'univers | 1000 |
| $T$ | Longueur de fenêtre (jours) | 504 |
| $F$ | Nombre de features par timestep | 2 (return + realized vol) |
| $K$ | Capacité latente (plafond) | 200 |
| $AU$ | Dimensions actives (auto-pruning) | Déterminé dynamiquement |
| $AU_{\max}^{\text{stat}}$ | Garde statistique | $\lfloor\sqrt{2 \cdot N_{\text{obs}} / r_{\min}}\rfloor$ |
| $D$ | Nombre d'éléments par fenêtre | $T \times F$ |
| $\sigma^2$ | Bruit d'observation (scalaire appris) | init 1.0, clamp [1e-4, 10] |
| $\gamma$ | Surpondération crise | 3.0 |
| $\lambda_{\text{co}}^{\max}$ | Poids max co-movement | 0.5 |
| $L$ | Profondeur encodeur (blocs résiduels) | $\max(3, \lceil\log_2(T/63)\rceil + 2)$ |
| $C_L$ | Largeur couche finale | $\max(384, \lceil 1.3 \times 2K \rceil)$ |
| $\lambda$ | Aversion au risque | 1.0 |
| $\alpha$ | Poids entropie | Coude de la frontière variance-entropie |

### Dépendances techniques

```
Python 3.11+
PyTorch >= 2.1
NumPy >= 1.24
SciPy >= 1.11
CVXPY >= 1.4 + MOSEK (ou ECOS fallback)
pandas >= 2.0
scikit-learn >= 1.3 (Ledoit-Wolf)
statsmodels >= 0.14 (tests statistiques)
pytest >= 7.0
```

---

## Topologie et décomposition modulaire

### Composants fonctionnels et couplage

```
Matrice de couplage (degré 0-4) :

                 data  vae   loss  train infer risk  optim wf    bench
data_pipeline    -     1     1     2     1     2     0     2     1
vae_architecture 1     -     3     3     2     0     0     0     0
loss_function    1     3     -     4     0     0     0     0     0
training         2     3     4     -     1     0     0     2     0
inference        1     2     0     1     -     3     0     1     0
risk_model       2     0     0     0     3     -     4     1     0
portfolio_optim  0     0     0     0     0     4     -     1     1
walk_forward     2     0     0     2     1     1     1     -     2
benchmarks       1     0     0     0     0     0     1     2     -
```

### Points de couplage critiques (degré ≥ 3)

| Paire | Degré | Invariant |
|-------|-------|-----------|
| vae_architecture ↔ loss_function | 3 (sémantique) | σ² scalaire (INV-002), D dans la loss (INV-001) |
| loss_function ↔ training | 4 (mathématique) | Modes P/F/A mutuellement exclusifs (INV-006), validation ELBO exclut γ et λ_co (INV-011), curriculum batching lié au curriculum λ_co (INV-010) |
| vae_architecture ↔ training | 3 (sémantique) | Reparameterization trick dans forward(), architecture détermine le training loop |
| inference ↔ risk_model | 3 (sémantique) | B_A shape (INV-003), convention AU filtering (KL > 0.01 nats) |
| risk_model ↔ portfolio_optim | 4 (mathématique) | Rescaling duale (INV-004), rotation principal factor basis (INV-007), gradient H (INV-009) |

### Décomposition en modules

| ID | Module | Composants | Dépendances | Densité contexte | Mode |
|----|--------|-----------|-------------|------------------|------|
| MOD-001 | `data_pipeline` | Data loading, returns, universe, windowing, z-scoring, VIX, crisis labels | — | Moyenne | teammate |
| MOD-002 | `vae_architecture` | build_vae.py, encoder, decoder, sizing rules | MOD-001 (I: shapes) | Haute | teammate |
| MOD-003 | `test_infrastructure` | Synthetic data, assertion framework, test fixtures | — | Faible | teammate |
| MOD-004 | `loss_function` | 3 modes, crisis weighting, co-movement loss, curriculum | MOD-002 (C) | Très haute | lead_session |
| MOD-005 | `training` | Training loop, batching, optimizer, early stopping, LR scheduler | MOD-004 (C), MOD-001 (D) | Très haute | lead_session |
| MOD-006 | `inference` | Composite profiles, aggregation, exposure matrix B | MOD-002 (C), MOD-005 (D: trained model) | Moyenne | subagent |
| MOD-007 | `risk_model` | AU filtering, rescaling, factor regression, Σ_z, D_ε, Σ_assets | MOD-006 (D) | Haute | subagent |
| MOD-008 | `portfolio_optimization` | Entropy, gradient, SCA, Armijo, cardinality, constraints | MOD-007 (D) | Très haute | lead_session |
| MOD-009 | `walk_forward` | Fold scheduling, Phase A/B, HP selection, metrics, holdout | MOD-001–008 (D) | Haute | lead_session |
| MOD-010 | `bench_equal_weight` | 1/N benchmark | MOD-001 (D), shared infra | Faible | teammate |
| MOD-011 | `bench_inverse_vol` | Inverse-volatility benchmark | MOD-001 (D) | Faible | teammate |
| MOD-012 | `bench_min_variance` | Minimum-variance Ledoit-Wolf | MOD-001 (D) | Moyenne | teammate |
| MOD-013 | `bench_erc` | Equal Risk Contribution (Spinu) | MOD-001 (D), MOD-012 (C: LW) | Moyenne | teammate |
| MOD-014 | `bench_pca_factor_rp` | PCA factor risk parity (Bai-Ng IC₂ + SCA) | MOD-001 (D), MOD-008 (C: SCA solver) | Haute | teammate |
| MOD-015 | `bench_pca_vol` | PCA + realized vol feature | MOD-014 (C) | Faible | teammate |
| MOD-016 | `integration` | E2E orchestration, reporting, statistical tests | Tous | Haute | lead_session |

### Graphe de dépendances (DAG)

```
Phase 1 (parallèle — Agent Team "infrastructure")
  MOD-001 (data_pipeline)
  MOD-002 (vae_architecture)
  MOD-003 (test_infrastructure)
      ↓ synchronisation

Phase 2 (séquentiel — Subagents builder-validator)
  MOD-004 (loss_function) ← MOD-002(C)
      ↓
  MOD-005 (training) ← MOD-004(C), MOD-001(D)
      ↓
  MOD-006 (inference) ← MOD-002(C), MOD-005(D)
      ↓
  MOD-007 (risk_model) ← MOD-006(D)
      ↓
  MOD-008 (portfolio_optimization) ← MOD-007(D)
      ↓ synchronisation

Phase 3 (parallèle — Agent Team "benchmarks")
  MOD-010 (bench_equal_weight) ← MOD-001(D)
  MOD-011 (bench_inverse_vol) ← MOD-001(D)
  MOD-012 (bench_min_variance) ← MOD-001(D)
  MOD-013 (bench_erc) ← MOD-012(C)
  MOD-014 (bench_pca_factor_rp) ← MOD-001(D), MOD-008(C: SCA)
  MOD-015 (bench_pca_vol) ← MOD-014(C)
      ↓ synchronisation

Phase 4 (séquentiel — lead session)
  MOD-009 (walk_forward) ← MOD-001–008(D)
      ↓
  MOD-016 (integration) ← Tous
```

### Structure du code

```
latent_risk_factors/
├── CLAUDE.md
├── pyproject.toml
├── docs/
│   ├── isd/
│   │   ├── 00_global.md          ← ce fichier
│   │   ├── 01_data_pipeline.md
│   │   ├── 02_vae_architecture.md
│   │   ├── ...
│   │   └── 16_integration.md
│   └── assertions/
│       └── contracts.yaml
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration centralisée (dataclasses)
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # CRSP / alternative data loading
│   │   ├── returns.py          # Log-return calculation
│   │   ├── universe.py         # Point-in-time universe construction
│   │   ├── windowing.py        # Sliding window + z-scoring
│   │   ├── crisis.py           # VIX threshold + crisis labeling
│   │   └── features.py         # Realized volatility + additional features
│   ├── vae/
│   │   ├── __init__.py
│   │   ├── build_vae.py        # Architecture construction (sizing rules)
│   │   ├── encoder.py          # InceptionHead + ResidualBlocks + projection
│   │   ├── decoder.py          # Transposed architecture
│   │   ├── model.py            # VAEModel (forward, reparameterize, loss modes)
│   │   └── loss.py             # Loss computation (3 modes, crisis weight, co-movement)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── batching.py         # Curriculum batching (synchronous + stratified / random)
│   │   ├── early_stopping.py   # Patience + best checkpoint
│   │   └── scheduler.py        # ReduceLROnPlateau wrapper
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── composite.py        # Sliding inference + aggregation → B
│   │   └── active_units.py     # AU measurement (KL > 0.01 nats)
│   ├── risk_model/
│   │   ├── __init__.py
│   │   ├── rescaling.py        # Dual rescaling (estimation + portfolio)
│   │   ├── factor_regression.py # Cross-sectional OLS → z_hat_t
│   │   ├── covariance.py       # Σ_z (Ledoit-Wolf), D_ε, Σ_assets assembly
│   │   └── conditioning.py     # Conditioning guard + ridge fallback
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── entropy.py          # H(w), ∇H(w), principal factor rotation
│   │   ├── sca_solver.py       # SCA + Armijo + multi-start
│   │   ├── constraints.py      # P_conc, P_turn, hard caps
│   │   ├── cardinality.py      # Sequential entropy-aware elimination
│   │   └── frontier.py         # Variance-entropy frontier + α calibration
│   ├── walk_forward/
│   │   ├── __init__.py
│   │   ├── folds.py            # Fold scheduling
│   │   ├── phase_a.py          # HP selection
│   │   ├── phase_b.py          # Deployment run
│   │   ├── metrics.py          # All metrics (3 layers)
│   │   └── selection.py        # Scoring + selection criterion
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract benchmark class
│   │   ├── equal_weight.py
│   │   ├── inverse_vol.py
│   │   ├── min_variance.py
│   │   ├── erc.py              # Spinu solver
│   │   ├── pca_factor_rp.py    # PCA + Bai-Ng IC₂ + SCA
│   │   └── pca_vol.py          # PCA + realized vol variant
│   └── integration/
│       ├── __init__.py
│       ├── pipeline.py         # End-to-end orchestrator
│       ├── statistical_tests.py # Wilcoxon, Holm-Bonferroni
│       └── reporting.py        # Results compilation
├── tests/
│   ├── unit/
│   │   ├── test_data_pipeline.py
│   │   ├── test_vae_architecture.py
│   │   ├── test_loss_function.py
│   │   ├── ...
│   │   └── test_benchmarks.py
│   ├── integration/
│   │   ├── test_vae_training.py
│   │   ├── test_risk_pipeline.py
│   │   ├── test_portfolio_pipeline.py
│   │   └── test_walk_forward.py
│   └── fixtures/
│       ├── synthetic_data.py   # Deterministic synthetic data generator
│       └── known_solutions.py  # Known analytical solutions for verification
└── scripts/
    ├── run_walk_forward.py
    └── run_benchmarks.py
```

---

## Assertions d'interface inter-modules

### MOD-001 → MOD-004/005 (data_pipeline → loss/training)

```python
def verify_data_pipeline_output(windows, crisis_labels, returns_df):
    """Assertions de sortie du data pipeline."""
    N, T, F = windows.shape
    assert T == 504, f"Window length {T} != 504"
    assert F == 2, f"Feature count {F} != 2"
    assert windows.dtype == torch.float32

    # Z-scored per window : mean ≈ 0, std ≈ 1 per feature
    for i in range(min(100, N)):
        for f in range(F):
            feat = windows[i, :, f]
            assert abs(feat.mean()) < 1e-5, f"Window {i} feature {f} mean {feat.mean():.6f} != 0"
            assert abs(feat.std() - 1.0) < 1e-3, f"Window {i} feature {f} std {feat.std():.6f} != 1"

    # Crisis labels : fraction in [0, 1]
    assert crisis_labels.shape == (N,)
    assert (crisis_labels >= 0).all() and (crisis_labels <= 1).all()

    # Returns are log-returns
    # Verification: sum of log-returns ≈ log(P_T/P_0)
    # (checked on raw returns before windowing, in data_pipeline tests)
```

### MOD-002 → MOD-004 (vae_architecture → loss_function)

```python
def verify_vae_forward(model, sample_input):
    """Assertions de compatibilité VAE ↔ loss."""
    x = sample_input  # (batch, T, F)
    x_hat, mu, log_var = model(x)

    assert x_hat.shape == x.shape, f"Reconstruction shape {x_hat.shape} != input {x.shape}"
    assert mu.shape[1] == model.K, f"mu dim {mu.shape[1]} != K={model.K}"
    assert log_var.shape == mu.shape

    # σ² is scalar
    assert model.log_sigma_sq.ndim == 0 or model.log_sigma_sq.numel() == 1
    sigma_sq = torch.exp(model.log_sigma_sq).item()
    assert 1e-4 <= sigma_sq <= 10, f"σ² = {sigma_sq} outside [1e-4, 10]"
```

### MOD-006 → MOD-007 (inference → risk_model)

```python
def verify_inference_output(B, AU, K, n_stocks, AU_max_stat):
    """Assertions de sortie de l'inférence."""
    assert B.shape == (n_stocks, K), f"B shape {B.shape} != ({n_stocks}, {K})"
    assert AU == (np.abs(B[:, :K]).sum(axis=0) > 1e-6).sum()  # Approximation

    # Active dimensions count
    assert 0 < AU <= K
    assert AU <= AU_max_stat, f"AU={AU} > AU_max_stat={AU_max_stat}"

    # B_A has correct shape after filtering
    B_A = B[:, active_dims]
    assert B_A.shape == (n_stocks, AU)
```

### MOD-007 → MOD-008 (risk_model → portfolio_optimization)

```python
def verify_risk_model_output(Sigma_z, D_eps, B_A_port, eigenvalues, V, n_stocks, AU):
    """Assertions de sortie du risk model."""
    assert Sigma_z.shape == (AU, AU)
    assert D_eps.shape == (n_stocks,)
    assert B_A_port.shape == (n_stocks, AU)

    # Σ_z is symmetric positive semi-definite
    assert np.allclose(Sigma_z, Sigma_z.T, atol=1e-10)
    assert all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues.min()}"

    # D_ε has floor
    assert (D_eps >= 1e-6).all(), f"D_eps min {D_eps.min()} < 1e-6"

    # V is orthogonal
    assert np.allclose(V @ V.T, np.eye(AU), atol=1e-8)

    # Rotated B preserves asset covariance
    B_prime = B_A_port @ V
    Lambda = np.diag(eigenvalues)
    Sigma_from_rotation = B_prime @ Lambda @ B_prime.T + np.diag(D_eps)
    Sigma_from_original = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)
    assert np.allclose(Sigma_from_rotation, Sigma_from_original, rtol=1e-6)
```

### MOD-008 output assertions (portfolio_optimization)

```python
def verify_portfolio_output(w, n_stocks, w_min=0.001, w_max=0.05):
    """Assertions de sortie de l'optimisation."""
    assert w.shape == (n_stocks,)
    assert abs(w.sum() - 1.0) < 1e-8, f"Weights sum {w.sum():.8f} != 1"
    assert (w >= -1e-10).all(), "Negative weight found"

    # Semi-continuous: w_i = 0 or w_i >= w_min
    active = w > 1e-10
    assert (w[active] >= w_min - 1e-8).all(), f"Active weight below w_min"
    assert (w <= w_max + 1e-8).all(), f"Weight exceeds w_max"
```

---


---

## Sections ISD — Phase 2 (séquentiel)

---

### MOD-004 — loss_function

**Phase :** 2 | **Mode :** lead_session | **Dépendances :** MOD-002 (C: VAEModel) | **Densité :** très haute
**Fichiers :** `src/vae/loss.py`, `tests/unit/test_loss_function.py`

#### Objectif

Implémenter les trois modes de calcul de la loss VAE (P/F/A), le mécanisme de crisis weighting, la co-movement loss, et le curriculum scheduling. Ce module est le cœur mathématique du VAE — toute erreur ici invalide silencieusement l'entraînement.

#### Entrées

| Nom | Type | Shape | Description |
|-----|------|-------|-------------|
| `x` | `torch.Tensor` | `(B, T, F)` | Fenêtres d'entrée z-scorées |
| `x_hat` | `torch.Tensor` | `(B, T, F)` | Reconstruction |
| `mu` | `torch.Tensor` | `(B, K)` | Moyenne encodeur |
| `log_var` | `torch.Tensor` | `(B, K)` | Log-variance encodeur |
| `log_sigma_sq` | `torch.Tensor` | scalar | Log σ² du modèle |
| `crisis_fractions` | `torch.Tensor` | `(B,)` | $f_c^{(w)}$ par fenêtre |
| `epoch` | int | — | Époque courante (pour le curriculum) |
| `mode` | str | — | "P", "F", ou "A" |

#### Sorties

| Nom | Type | Description |
|-----|------|-------------|
| `total_loss` | `torch.Tensor` | scalar, loss totale pour backprop |
| `loss_components` | dict | recon, kl, co_mov, sigma_sq (pour monitoring) |

#### Sous-tâche 1 : Loss reconstruction pondérée par crise

```
MSE(w) = (1/(T×F)) · Σ_{t,f} (x_{w,t,f} - x̂_{w,t,f})²    [per-element mean]

γ_eff(w) = 1 + f_c(w) · (γ - 1)                             [continuous crisis weight]

L_recon_weighted = (1/|B|) · Σ_w γ_eff(w) · MSE(w)          [batch mean, weighted]
```

**ATTENTION :** MSE est une moyenne per-element (divisée par T×F), pas une somme. Le facteur D = T×F est appliqué séparément comme coefficient multiplicatif.

#### Sous-tâche 2 : Loss KL

$$\mathcal{L}_{\text{KL}} = \frac{1}{N_{\text{batch}}} \sum_{i=1}^{N_{\text{batch}}} \frac{1}{2} \sum_{k=1}^{K} \left( \mu_{ik}^2 + \exp(\log\_var_{ik}) - \log\_var_{ik} - 1 \right)$$

La KL est moyennée sur le batch (dimension 0) et sommée sur les dimensions latentes (dimension 1). Le 1/2 est à l'extérieur de la somme sur k.

#### Sous-tâche 3 : Assemblage — Mode P (primary)

$$\mathcal{L} = \frac{D}{2\sigma^2} \cdot \mathcal{L}_{\text{recon, weighted}} + \frac{D}{2}\ln\sigma^2 + \mathcal{L}_{\text{KL}} + \lambda_{\text{co}}(t) \cdot \mathcal{L}_{\text{co-mov}}$$

Où :
- $D = T \times F$ (pour T=504, F=2 : D=1008)
- $\sigma^2 = \text{clamp}(\exp(\log\_sigma\_sq), 10^{-4}, 10)$
- $\beta = 1$ fixe (NE PAS modifier)
- $\lambda_{\text{co}}(t)$ suit le curriculum (sous-tâche 5)

Le terme $(D/2)\ln\sigma^2$ est le log-normalisation de la Gaussienne. Il pénalise $\sigma^2$ élevé (le modèle ne peut pas "tricher" en augmentant σ² pour réduire le coût de reconstruction).

#### Sous-tâche 4 : Assemblage — Mode F (fallback)

$$\mathcal{L}_t = \frac{D}{2} \cdot \mathcal{L}_{\text{recon, weighted}} + \beta_t \cdot \mathcal{L}_{\text{KL}} + \lambda_{\text{co}}(t) \cdot \mathcal{L}_{\text{co-mov}}$$

Où :
- $\sigma^2 = 1$ GELÉ (pas de gradient sur log_sigma_sq)
- $D/2$ est RETENU (scaling dimensionnel)
- $\beta_t = \min(1, t / T_{\text{warmup}})$, rampe linéaire
- $T_{\text{warmup}}$ : 10-30% des epochs totaux

**CRUCIAL :** NE PAS retirer D/2. Sans D/2, MSE ≈ 0.3-0.7 serait dominé par KL ≈ 60-120 nats → posterior collapse immédiat.

#### Sous-tâche 5 : Assemblage — Mode A (advanced)

$$\mathcal{L} = \frac{D}{2\sigma^2} \cdot \mathcal{L}_{\text{recon, weighted}} + \frac{D}{2}\ln\sigma^2 + \beta \cdot \mathcal{L}_{\text{KL}} + \lambda_{\text{co}}(t) \cdot \mathcal{L}_{\text{co-mov}}$$

Identique au Mode P mais avec $\beta > 1$ (fixe, pas annealé). Range : β ∈ [1.0, 4.0].

#### Sous-tâche 6 : Co-movement loss

```python
def compute_co_movement_loss(
    mu: torch.Tensor,           # (B, K) latent means
    window_metadata: ...,       # stock_ids + dates for pairing
    returns_data: ...,          # raw returns for Spearman computation
    max_pairs: int = 2048,
    delta_sync: int = 21,       # max date gap for synchronization
) -> torch.Tensor:
    """
    Pour chaque paire (i, j) éligible dans le batch :
      1. Vérifier : stocks distincts, |end_date_i - end_date_j| ≤ δ_sync,
         ≥ 80% données valides dans la période commune
      2. ρ_ij = Spearman rank correlation sur returns BRUTS (pas z-scorés)
         dans le segment temporel commun
      3. d(z_i, z_j) = cosine distance = 1 - cos_sim(μ_i, μ_j)
      4. g(ρ_ij) = 1 - ρ_ij  (target distance)
      5. L_co = (1/|P|) · Σ (d(z_i, z_j) - g(ρ_ij))²

    Subsample to max_pairs if |P| > max_pairs.
    """
```

**Note :** Le calcul de la corrélation Spearman est coûteux dans le forward pass. Pré-calculer les corrélations par paires pour les fenêtres synchrones et les stocker dans un lookup table, mis à jour à chaque fold.

#### Sous-tâche 7 : Curriculum scheduling

```python
def get_lambda_co(epoch: int, total_epochs: int, lambda_co_max: float = 0.5) -> float:
    """
    Phase 1 (0 → 30% epochs) : λ_co = λ_co_max
    Phase 2 (30% → 60% epochs) : λ_co décroît linéairement de λ_co_max à 0
    Phase 3 (60% → 100% epochs) : λ_co = 0

    Returns: lambda_co for this epoch
    """
    phase1_end = int(0.30 * total_epochs)
    phase2_end = int(0.60 * total_epochs)

    if epoch < phase1_end:
        return lambda_co_max
    elif epoch < phase2_end:
        progress = (epoch - phase1_end) / (phase2_end - phase1_end)
        return lambda_co_max * (1.0 - progress)
    else:
        return 0.0
```

#### Sous-tâche 8 : Validation ELBO

```python
def compute_validation_elbo(x, x_hat, mu, log_var, log_sigma_sq, T, F):
    """
    ELBO de validation — EXCLUT γ et λ_co, INCLUT σ².

    L_val = D/(2σ²) · L_recon^(γ=1) + (D/2)·ln(σ²) + L_KL

    Où L_recon^(γ=1) est le MSE moyen NON PONDÉRÉ par crise.
    """
```

#### Invariants applicables

- **INV-001 :** D = T × F apparaît comme coefficient de L_recon. Vérifier : `assert recon_coeff == T * F / (2 * sigma_sq)`.
- **INV-002 :** σ² est scalaire. `assert log_sigma_sq.ndim == 0`.
- **INV-006 :** Modes mutuellement exclusifs. `assert not (mode == 'P' and beta != 1.0)`.
- **INV-010 :** Le curriculum λ_co retourne 0 en Phase 3.
- **INV-011 :** Validation ELBO exclut γ et λ_co.

#### Pièges connus

- **NE PAS** omettre le facteur D = T × F. C'est l'erreur la plus critique de tout le pipeline. Sans D, la reconstruction (~0.3-0.7 nats) est écrasée par la KL (~60-120 nats), le modèle collapse immédiatement.
- **NE PAS** confondre MSE per-element (correct) avec MSE sum (incorrect). Le D est appliqué comme coefficient multiplicatif, pas intégré dans le MSE.
- **NE PAS** combiner σ² appris avec β annealing (Mode P + Mode F simultanés).
- **NE PAS** inclure γ dans la validation ELBO.
- **NE PAS** inclure λ_co dans la validation ELBO.
- **NE PAS** calculer la Spearman correlation sur les données z-scorées — utiliser les returns bruts.

#### Tests requis

1. `test_D_factor_present` : loss_recon_coeff == T*F / (2*sigma_sq) pour Mode P
2. `test_mode_P_gradients` : σ² reçoit un gradient ; β est fixe à 1
3. `test_mode_F_sigma_frozen` : σ² n'a pas de gradient, D/2 est retenu
4. `test_mode_F_beta_annealing` : β_t = min(1, t/T_warmup) correctement calculé
5. `test_mode_A_beta_applied` : KL multiplié par β > 1
6. `test_modes_exclusive` : erreur si mode invalide ou combinaison interdite
7. `test_crisis_weight_gamma_1` : γ=1 → γ_eff=1 pour toutes les fenêtres
8. `test_crisis_weight_gamma_3` : γ=3, f_c=1 → γ_eff=3 ; f_c=0 → γ_eff=1
9. `test_curriculum_phases` : λ_co correcte à chaque phase boundary
10. `test_validation_elbo_excludes_gamma` : même L_val pour γ=1 et γ=3
11. `test_co_movement_symmetric` : L_co(i,j) == L_co(j,i)
12. `test_loss_finite` : pas de NaN ou Inf pour inputs valides

---

### MOD-005 — training

**Phase :** 2 | **Mode :** lead_session | **Dépendances :** MOD-004 (C), MOD-001 (D: windows) | **Densité :** très haute
**Fichiers :** `src/training/*.py`, `tests/unit/test_training.py`

#### Objectif

Implémenter la boucle d'entraînement complète avec curriculum batching (synchrone+stratifié pour co-movement phases, random pour free refinement), early stopping sur validation ELBO, ReduceLROnPlateau, et le protocole de sauvegarde des checkpoints.

#### Sous-tâche 1 : Curriculum batching (batching.py)

```python
class CurriculumBatchSampler:
    """
    Phases 1-2 (λ_co > 0) : SYNCHRONOUS + STRATIFIED batching
      - Sélectionner un bloc temporel aléatoire (δ_sync = 21 jours)
      - Pré-clusterer les stocks en S strates (10-20 groupes, k-means sur
        trailing 63j returns, ou GICS sectors comme proxy zero-cost)
      - Sampler B/S fenêtres par strate
      - Garantir synchronisation temporelle pour co-movement loss

    Phase 3 (λ_co = 0) : RANDOM SHUFFLING standard
      - Fenêtres tirées uniformément sur tous les stocks et périodes
    """
```

**Transition :** Le sampler change de stratégie au moment exact où λ_co atteint 0 (début de Phase 3 du curriculum).

#### Sous-tâche 2 : Training loop (trainer.py)

```python
class VAETrainer:
    def __init__(self, model, optimizer, loss_mode, config):
        """
        optimizer: Adam(model.parameters(), lr=η_0, betas=(0.9, 0.999), eps=1e-8)
        weight_decay: 1e-5 (on all weights)
        """

    def train_epoch(self, train_loader, epoch, total_epochs) -> dict:
        """
        Un epoch :
        1. Déterminer λ_co via curriculum (get_lambda_co)
        2. Déterminer le type de batching (synchrone si λ_co > 0, random sinon)
        3. Pour chaque batch :
           a. Forward pass
           b. Compute loss (mode P/F/A selon config)
           c. Backward + optimizer step
           d. Clamp log_sigma_sq pour assurer σ² ∈ [1e-4, 10] :
              with torch.no_grad():
                model.log_sigma_sq.clamp_(math.log(1e-4), math.log(10))
        4. Collecter les métriques : loss_total, loss_recon, loss_kl,
           loss_co, sigma_sq, mse_crisis, mse_normal

        Returns: dict of epoch-level metrics
        """

    def validate(self, val_loader) -> float:
        """
        Compute validation ELBO (INV-011 : exclut γ et λ_co, inclut σ²).
        Returns: validation_elbo (lower is better)
        """

    def fit(self, train_dataset, val_dataset, max_epochs=100) -> dict:
        """
        Boucle complète :
        1. Pour chaque epoch 1..max_epochs :
           a. train_epoch()
           b. validate() → val_elbo
           c. scheduler.step(val_elbo)  [SAUF si Mode F et epoch < T_warmup]
           d. early_stopping.check(val_elbo) [SAUF si Mode F et epoch < T_warmup]
           e. Si early stopping triggered → restore best weights → stop
        2. Retourner : best_epoch (E*), best_val_elbo, training_history

        Mode F spécifique :
        - Scheduler et early stopping DÉSACTIVÉS pendant T_warmup
        - Activés uniquement une fois β_t = 1.0
        """
```

#### Sous-tâche 3 : Early stopping (early_stopping.py)

```python
class EarlyStopping:
    """
    Patience: 10 epochs sans amélioration de validation ELBO.
    Restore best weights: oui.
    Record E* : l'epoch du meilleur checkpoint.
    """
```

#### Sous-tâche 4 : Monitoring

Métriques à logger à chaque epoch :
- `train_loss`, `val_elbo`
- `sigma_sq` (valeur courante)
- `AU` (nombre de dimensions avec KL_k > 0.01 nats, calculé sur le batch)
- `mse_crisis / mse_normal` (ratio, target [0.5, 2.0])
- `effective_eta` (contribution effective des crises)
- `lambda_co` (valeur courante du curriculum)
- `learning_rate` (valeur courante)

**Diagnostic overfit :** si val_elbo_best / train_loss_best < 0.85 ou > 1.5, flag le fold.

#### Invariants applicables

- **INV-010 :** Batching synchrone en Phases 1-2, random en Phase 3.
- **INV-011 :** Validation ELBO exclut γ et λ_co.
- **INV-005 :** Training sur données ≤ training_end_date uniquement.
- **INV-006 :** Scheduler et early stopping désactivés pendant warmup (Mode F).

#### Pièges connus

- **NE PAS** oublier de clamper log_sigma_sq après chaque optimizer step — sans clamp, σ² peut diverger.
- **NE PAS** activer le scheduler pendant le warmup de Mode F — les changements mécaniques de loss triggèrent des plateaux artificiels.
- **NE PAS** utiliser le batching random pendant les phases de co-movement — la co-movement loss n'est pas calculable sans synchronisation temporelle.
- **NE PAS** oublier de passer en mode `model.eval()` + `torch.no_grad()` pour la validation.

#### Tests requis

1. `test_sigma_sq_clamped` : σ² reste dans [1e-4, 10] après 100 steps avec données aléatoires
2. `test_curriculum_batching_transition` : batching change au bon moment
3. `test_early_stopping_patience` : stop après 10 epochs sans amélioration
4. `test_best_checkpoint_restored` : poids du meilleur epoch restaurés
5. `test_mode_F_warmup_protection` : scheduler/early_stopping inactifs pendant warmup
6. `test_training_loss_decreases` : loss diminue sur 5 epochs (données synthétiques simples)

---

### MOD-006 — inference

**Phase :** 2 | **Mode :** subagent | **Dépendances :** MOD-002 (C), MOD-005 (D: trained model) | **Densité :** moyenne
**Fichiers :** `src/inference/*.py`, `tests/unit/test_inference.py`

#### Objectif

Passer toutes les fenêtres de chaque stock à travers l'encodeur entraîné (forward pass sans sampling), agréger les vecteurs latents locaux en profils composites, construire la matrice d'exposition B (n × K), et mesurer AU.

#### Sous-tâche 1 : Inference stride-1

```python
def infer_latent_trajectories(
    model: VAEModel,
    windows: torch.Tensor,         # (N_windows, T, F)
    window_metadata: pd.DataFrame,  # stock_id, start_date, end_date
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """
    Forward pass (encode only, pas de sampling) pour toutes les fenêtres.

    model.eval() + torch.no_grad() — inférence uniquement.
    Utilise model.encode(x) qui retourne mu directement.

    Returns: dict mapping stock_id → ndarray of shape (n_windows_for_stock, K)
    """
```

#### Sous-tâche 2 : Agrégation → profils composites

```python
def aggregate_profiles(
    trajectories: dict[str, np.ndarray],  # stock_id → (n_windows, K)
    method: str = "mean",
) -> np.ndarray:
    """
    Agrège les vecteurs latents locaux en profils composites.

    Default : mean (toutes les fenêtres contribuent également,
    préservant la mémoire de tous les régimes historiques).

    Returns: B of shape (n_stocks, K)
    """
```

#### Sous-tâche 3 : Mesure AU (active_units.py)

```python
def measure_active_units(
    model: VAEModel,
    windows: torch.Tensor,
    batch_size: int = 512,
) -> tuple[int, np.ndarray, list[int]]:
    """
    Calcule la KL marginale par dimension :
      KL_k = (1/N) Σ_i (1/2)(μ²_ik + exp(log_var_ik) - log_var_ik - 1)

    Active unit k ⟺ KL_k > 0.01 nats.

    AU_max_stat = floor(sqrt(2 × N_obs / r_min))
    avec N_obs = nombre de jours d'historique, r_min = 2.

    Si AU > AU_max_stat, tronquer A aux AU_max_stat dimensions
    avec la KL marginale la plus élevée.

    Returns:
      AU: int (nombre de dimensions actives, possiblement tronqué)
      kl_per_dim: ndarray (K,) — KL marginale par dimension
      active_dims: list[int] — indices des dimensions actives
    """
```

#### Invariants applicables

- **INV-003 :** B_A.shape == (n_stocks, AU), AU ≤ AU_max_stat.
- **CONV-07 :** AU = |{k : KL_k > 0.01 nats}|, déterminé une fois par retraining.
- **INV-005 :** Inférence utilise uniquement des fenêtres du training set.

#### Pièges connus

- **NE PAS** utiliser `model.forward()` (qui sample z) — utiliser `model.encode()` (qui retourne mu directement). L'inférence doit être déterministe.
- **NE PAS** oublier `model.eval()` et `torch.no_grad()` — sinon le dropout est actif et les résultats ne sont pas reproductibles.
- **NE PAS** oublier la troncation statistique : si AU > AU_max_stat, garder uniquement les AU_max_stat dimensions avec KL_k le plus élevé.

#### Tests requis

1. `test_inference_deterministic` : deux passes identiques sur les mêmes données
2. `test_B_shape` : B.shape == (n_stocks, K)
3. `test_AU_measurement` : AU correct sur données synthétiques (facteurs connus)
4. `test_AU_truncation` : si AU > AU_max_stat, troncation correcte
5. `test_active_dims_ordering` : dimensions triées par KL décroissante

---

### MOD-007 — risk_model

**Phase :** 2 | **Mode :** subagent | **Dépendances :** MOD-006 (D) | **Densité :** haute
**Fichiers :** `src/risk_model/*.py`, `tests/unit/test_risk_model.py`

#### Objectif

Transformer la matrice d'exposition B_A (shape, sans échelle) en un modèle de risque factoriel complet : rescaling dual, estimation des facteurs par OLS cross-sectionnelle, estimation de Σ_z (Ledoit-Wolf), estimation de D_ε, assemblage de Σ_assets.

#### Sous-tâche 1 : Rescaling dual (rescaling.py)

**Pour l'estimation historique** (date-specific) :

$$\tilde{B}_{\mathcal{A},i,t} = \frac{\sigma_{i,t}}{\bar{\sigma}_t} \cdot \bar{\mu}_{\mathcal{A},i}$$

- $\sigma_{i,t}$ : trailing 252j annualized vol du stock i à la date t (de MOD-001)
- $\bar{\sigma}_t$ : médiane cross-sectionnelle de $\sigma_{i,t}$ sur l'univers $\mathcal{U}_t$
- $\bar{\mu}_{\mathcal{A},i}$ : profil composite filtré aux AU dimensions actives

**Ratio bounding :** $R_{i,t} = \sigma_{i,t} / \bar{\sigma}_t$ est winsorisé cross-sectionnellement à chaque date aux percentiles $[P_5(R_{\cdot,t}), P_{95}(R_{\cdot,t})]$.

**Pour la construction de portefeuille** (current-date) :

$$\tilde{B}_{\mathcal{A},i}^{\text{port}} = \frac{\sigma_{i,\text{now}}}{\bar{\sigma}_{\text{now}}} \cdot \bar{\mu}_{\mathcal{A},i}$$

Même formule mais avec les volatilités à la date de rebalancement uniquement. Même winsorisation.

```python
def rescale_estimation(
    B_A: np.ndarray,           # (n, AU) shape-only exposures
    trailing_vol: pd.DataFrame, # (n_dates, n_stocks) σ_{i,t}
    universe_snapshots: dict,   # date → list of active stocks
    percentile_bounds: tuple = (5, 95),
) -> dict[str, np.ndarray]:
    """
    Returns: dict mapping date_str → B_A_t of shape (n_active_t, AU)
    Each B_A_t uses only stocks active at that date.
    """

def rescale_portfolio(
    B_A: np.ndarray,
    trailing_vol: pd.DataFrame,
    current_date: pd.Timestamp,
    universe: list[str],
    percentile_bounds: tuple = (5, 95),
) -> np.ndarray:
    """Returns: B_A_port of shape (n, AU)"""
```

#### Sous-tâche 2 : Régression cross-sectionnelle (factor_regression.py)

$$\hat{z}_t = (\tilde{B}_{\mathcal{A},t}^T \tilde{B}_{\mathcal{A},t})^{-1} \tilde{B}_{\mathcal{A},t}^T r_t$$

OLS à chaque date t, en utilisant uniquement les stocks actifs à cette date.

```python
def estimate_factor_returns(
    B_A_by_date: dict[str, np.ndarray],  # date → (n_active, AU)
    returns: pd.DataFrame,                # (n_dates, n_stocks) log-returns
) -> np.ndarray:
    """
    Returns: z_hat of shape (n_dates, AU)
    """
```

**Conditioning guard :** si $\kappa(\tilde{B}_{\mathcal{A},t}^T \tilde{B}_{\mathcal{A},t}) > 10^6$, appliquer un ridge minimal : $\lambda = 10^{-6} \cdot \text{tr}(\tilde{B}_{\mathcal{A},t}^T \tilde{B}_{\mathcal{A},t}) / \text{AU}$.

#### Sous-tâche 3 : Estimation Σ_z et D_ε (covariance.py)

**Σ_z :** Covariance empirique de $\{\hat{z}_t\}$ sur l'historique COMPLET + shrinkage Ledoit-Wolf :

$$\hat{\Sigma}_z = (1 - \delta^*) S_{\text{emp}} + \delta^* \frac{\text{tr}(S_{\text{emp}})}{\text{AU}} I_{\text{AU}}$$

$\delta^*$ calculé analytiquement (utiliser `sklearn.covariance.LedoitWolf` ou implémentation directe).

**D_ε :** Variances idiosyncratiques à partir des résidus :

$$\varepsilon_{i,t} = r_{i,t} - \tilde{B}_{\mathcal{A},i,t} \hat{z}_t$$

$D_{\varepsilon,i} = \text{Var}(\varepsilon_{i,\cdot})$ sur toutes les dates où stock i est actif. Floor : $D_{\varepsilon,i} \geq 10^{-6}$.

**ATTENTION :** Les résidus utilisent la rescaling DATE-SPECIFIC $\tilde{B}_{\mathcal{A},i,t}$, pas la rescaling portfolio.

#### Sous-tâche 4 : Assemblage Σ_assets et eigendecomposition

$$\Sigma_{\text{assets}} = \tilde{B}_{\mathcal{A}}^{\text{port}} \cdot \Sigma_z \cdot (\tilde{B}_{\mathcal{A}}^{\text{port}})^T + D_\varepsilon$$

Eigendecomposition de Σ_z : $\Sigma_z = V \Lambda V^T$, avec $\Lambda = \text{diag}(\lambda_1, ..., \lambda_{\text{AU}})$.

Rotation de B_A_port dans la base des facteurs principaux :

$$\tilde{B}'^{\text{port}}_{\mathcal{A}} = \tilde{B}_{\mathcal{A}}^{\text{port}} \cdot V$$

```python
def assemble_risk_model(
    B_A_port: np.ndarray,     # (n, AU)
    Sigma_z: np.ndarray,      # (AU, AU)
    D_eps: np.ndarray,        # (n,)
) -> dict:
    """
    Returns dict with:
      Sigma_assets: (n, n)
      eigenvalues: (AU,) — λ_k, sorted descending
      V: (AU, AU) — eigenvectors
      B_prime_port: (n, AU) — rotated exposures
    """
```

#### Invariants applicables

- **INV-003 :** B_A.shape == (n, AU), AU ≤ AU_max_stat.
- **INV-004 :** Rescaling date-specific pour estimation, current-date pour portefeuille.
- **INV-007 :** Eigenvalues de Σ_z ≥ 0 (PSD). La rotation préserve Σ_assets.
- **INV-008 :** Ratio σ_i,t / σ_bar_t winsorisé [P5, P95].

#### Pièges connus

- **NE PAS** utiliser la rescaling portfolio (current-date) pour l'estimation historique — cela attribue incorrectement le risque systématique aux résidus.
- **NE PAS** oublier le floor D_ε ≥ 1e-6 — sans floor, un stock à variance idiosyncratique nulle reçoit un poids infini dans l'optimisation.
- **NE PAS** calculer Σ_z sur une fenêtre rolling — utiliser l'historique COMPLET (principe anti-cyclique).
- **NE PAS** oublier la conditioning guard sur B_A_t^T B_A_t.

#### Tests requis

1. `test_rescaling_dual` : estimation vs portfolio rescaling produisent des résultats différents
2. `test_winsorization` : ratios bornés à [P5, P95]
3. `test_factor_regression_identity` : pour B = I et r = z, z_hat ≈ z
4. `test_Sigma_z_psd` : toutes les eigenvalues ≥ 0
5. `test_D_eps_floor` : min(D_eps) ≥ 1e-6
6. `test_covariance_reconstruction` : B'ΛB'^T + D_ε == BΣ_zB^T + D_ε
7. `test_conditioning_guard` : ridge appliqué quand κ > 1e6
8. `test_rotation_preserves_covariance` : Σ_assets identique avant et après rotation

---

### MOD-008 — portfolio_optimization

**Phase :** 2 | **Mode :** lead_session | **Dépendances :** MOD-007 (D) | **Densité :** très haute
**Fichiers :** `src/portfolio/*.py`, `tests/unit/test_portfolio_optimization.py`

#### Objectif

Implémenter l'optimisation de portefeuille par entropie factorielle : calcul de H(w) et ∇H(w) dans la base des facteurs principaux, solver SCA avec Armijo backtracking, multi-start (5 initialisations), cardinality enforcement par élimination séquentielle entropy-aware, et calibration de α via la frontière variance-entropie.

#### Sous-tâche 1 : Entropie et gradient (entropy.py)

**Entropie :**

$$H(w) = -\sum_{k=1}^{\text{AU}} \hat{c}'_k \ln \hat{c}'_k$$

Où :
- $B' = \tilde{B}'^{\text{port}}_{\mathcal{A}}$ (rotated, shape n × AU)
- $\beta' = B'^T w$ (portfolio exposure, shape AU)
- $c'_k = (\beta'_k)^2 \cdot \lambda_k$ (risk contribution from principal factor k, toujours ≥ 0)
- $C = \sum_k c'_k$ (total systematic risk)
- $\hat{c}'_k = c'_k / C$ (normalized risk contributions)

**Gradient :**

$$\nabla_w H = -\frac{2}{C}\; B'\; \phi, \qquad \phi_k = \lambda_k \, \beta'_k \,(\ln \hat{c}'_k + H)$$

**Vérification :** Au maximum d'entropie ($\hat{c}'_k = 1/\text{AU}$ ∀k), $\ln \hat{c}'_k = -\ln(\text{AU}) = -H$, donc $\phi_k = 0$ et $\nabla_w H = 0$.

```python
def compute_entropy_and_gradient(
    w: np.ndarray,              # (n,)
    B_prime: np.ndarray,        # (n, AU)
    eigenvalues: np.ndarray,    # (AU,)
    eps: float = 1e-30,         # pour la stabilité numérique de ln
) -> tuple[float, np.ndarray]:
    """
    Returns: (H, grad_H) where H is scalar and grad_H is (n,)

    Numerical stability: if c'_k < eps, set ln(c'_k) = ln(eps)
    and exclude from normalization to avoid NaN.
    """
```

#### Sous-tâche 2 : SCA solver (sca_solver.py)

**Multi-start (M=5) :**
1. Equal-weight : w_i = 1/n
2. Minimum variance : w* = argmin w^T Σ w (QP convexe)
3. Approximate ERC (Spinu) : w ∝ solution de $w_i (Σw)_i = \text{const}$
4-5. Random : cardinality n_r ~ U(30, 300), stocks uniformes, poids Dirichlet(1), projection sur contraintes (clip w_max, renormalize, zero < w_min, renormalize, itérer 2-3 passes)

**Itérations SCA :**
```
Pour chaque start m = 1..M :
  w = w_init_m
  Pour t = 1..max_iter :
    1. Lineariser H(w) autour de w^(t) : H_lin(w) = H(w^(t)) + ∇H(w^(t))^T (w - w^(t))
    2. Résoudre le sous-problème convexe (CVXPY) :
       w* = argmax [ w^T μ - λ w^T Σ w + α · ∇H^T w - φ P_conc(w) - P_turn(w, w_old) ]
       s.t. w_i ∈ [0, w_max], 1^T w = 1, ||w - w_old||_1 / 2 ≤ τ_max
       (Note : le terme α · ∇H^T w est LINÉAIRE en w — c'est ça qui rend le sous-problème convexe)
    3. Armijo backtracking pour trouver η_step :
       η = max{ρ^j : j=0,...,j_max} tel que
       f(w^(t) + η(w* - w^(t))) ≥ f(w^(t)) + c · η · Δ_surr
       avec c=1e-4, ρ=0.5, j_max=20
       Δ_surr = f_surr(w*) - f_surr(w^(t)) ≥ 0
    4. Update : w^(t+1) = w^(t) + η(w* - w^(t))
    5. Convergence : |f(w^(t+1)) - f(w^(t))| < 1e-8

  Appliquer cardinality enforcement (sous-tâche 3)
  Stocker (w_final_m, f_final_m, H_final_m)

Sélectionner m* = argmax_m f(w_final_m)
```

**Objectif f :**
- Default (μ=0) : $f = -\lambda w^T \Sigma w + \alpha H(w) - \phi P_{\text{conc}}(w) - P_{\text{turn}}(w, w^{\text{old}})$
- Directional : $f = w^T \mu - \lambda w^T \Sigma w + \alpha H(w) - \phi P_{\text{conc}}(w) - P_{\text{turn}}(w, w^{\text{old}})$

#### Sous-tâche 3 : Cardinality enforcement (cardinality.py)

```python
def enforce_cardinality(
    w: np.ndarray,
    w_min: float,
    sca_solver_fn,        # callable to re-optimize on reduced active set
    B_prime, eigenvalues,  # for entropy computation
    max_eliminations: int = 100,
) -> np.ndarray:
    """
    Répéter :
    1. S_sub = {i : 0 < w_i < w_min}. Si vide → stop.
    2. Pour chaque i ∈ S_sub : ΔH_i = H(w) - H(w^(-i)) (coût entropique d'élimination)
    3. Éliminer i* = argmin ΔH_i (le moins coûteux)
    4. Re-optimiser via SCA sur l'active set réduit
    5. Retour à l'étape 1

    Garantie de convergence : active set diminue strictement à chaque itération.
    En pratique : 5-15 itérations pour n=1000.
    """
```

#### Sous-tâche 4 : Contraintes (constraints.py)

**Concentration penalty :**
$$P_{\text{conc}}(w) = \sum_i \max(0, w_i - \bar{w})^2, \quad \bar{w} = 3\%$$

**Turnover penalty (Almgren-Chriss) :**
$$P_{\text{turn}}(w, w^{\text{old}}) = \kappa_1 \cdot \frac{1}{2}\sum_i |w_i - w_i^{\text{old}}| + \kappa_2 \cdot \sum_i \max(0, |w_i - w_i^{\text{old}}| - \bar{\delta})^2$$

Premier rebalancement : κ₁ = κ₂ = 0.

**Hard constraints :**
- $w_i \geq 0$ (long-only)
- $\sum_i w_i = 1$ (fully invested)
- $w_i \leq w_{\max}^{\text{hard}} = 5\%$
- $\frac{1}{2}\|w - w^{\text{old}}\|_1 \leq \tau_{\max}^{\text{hard}} = 30\%$

#### Sous-tâche 5 : Frontière variance-entropie (frontier.py)

```python
def compute_variance_entropy_frontier(
    Sigma_assets, B_prime, eigenvalues, D_eps,
    lambda_risk: float = 1.0,
    alpha_grid: list = [0, 0.01, 0.05, 0.1, 0.5, 1, 5],
    **constraint_params,
) -> pd.DataFrame:
    """
    Pour chaque α, résoudre l'optimisation default (μ=0).
    Retourner DataFrame avec colonnes: alpha, variance, entropy, n_active_positions.

    Le coude (elbow) de la frontière = operating point.
    Sélection automatique : α où ΔH/ΔVar < threshold.
    """
```

#### Invariants applicables

- **INV-007 :** H(w) calculé dans la base des facteurs principaux. c'_k ≥ 0 ∀k.
- **INV-009 :** Gradient exactement comme spécifié. ∇H = 0 au maximum.
- **INV-012 :** Contraintes identiques pour VAE et benchmarks.

#### Pièges connus

- **NE PAS** calculer l'entropie dans la base latente brute — les contributions c_k peuvent être négatives si Σ_z n'est pas diagonale. Toujours utiliser la base des facteurs principaux (après rotation V).
- **NE PAS** oublier le terme constant α · H(w^(t)) dans la linéarisation — il n'affecte pas w* mais il est nécessaire pour le calcul de Δ_surr dans Armijo.
- **NE PAS** utiliser un full step (η=1) sans Armijo — en early iterations, le full step peut DIMINUER l'objectif réel.
- **NE PAS** oublier la condition de convergence |f(w^(t+1)) - f(w^(t))| < 1e-8, pas |w^(t+1) - w^(t)|.
- **NE PAS** re-optimiser les paramètres de contrainte (φ, κ₁, κ₂) par benchmark — ils sont fixés une fois pour le VAE et appliqués identiquement à tous les modèles (INV-012).

#### Tests requis

1. `test_entropy_gradient_at_maximum` : ∇H = 0 quand ĉ'_k = 1/AU ∀k
2. `test_entropy_gradient_numerical` : gradient analytique ≈ gradient numérique (finite diff)
3. `test_sca_convergence` : convergence en < 100 iterations sur problème simple
4. `test_armijo_sufficient_decrease` : f(w_{t+1}) ≥ f(w_t) à chaque iteration
5. `test_multi_start_deterministic` : mêmes résultats avec même seed
6. `test_cardinality_enforcement` : aucun w_i dans (0, w_min) après enforcement
7. `test_constraints_satisfied` : tous les hard constraints respectés
8. `test_turnover_first_rebalancing` : κ₁ = κ₂ = 0 au premier rebalancement
9. `test_known_solution` : pour Σ diagonal et B = I, solution analytique retrouvée


---

## Sections ISD — Phase 3 (parallèle : benchmarks)

---

### MOD-010 à MOD-015 — Benchmarks

**Phase :** 3 | **Mode :** teammates (Agent Team "benchmarks") | **Densité :** faible à haute
**Fichiers :** `src/benchmarks/*.py`, `tests/unit/test_benchmarks.py`

Tous les benchmarks héritent d'une classe abstraite commune et partagent l'infrastructure de MOD-001 (data_pipeline) et les contraintes de portefeuille de MOD-008.

#### Classe abstraite commune (base.py)

```python
from abc import ABC, abstractmethod

class BenchmarkModel(ABC):
    """
    Interface commune pour tous les benchmarks.

    Chaque benchmark reçoit exactement les mêmes inputs (univers, returns,
    contraintes) et produit exactement le même format de sortie (poids).
    """

    def __init__(self, constraint_params: dict):
        """
        constraint_params: w_max, w_min, phi, kappa_1, kappa_2,
        delta_bar, tau_max, lambda_risk (identiques au VAE — INV-012).
        """
        self.constraint_params = constraint_params

    @abstractmethod
    def fit(self, returns: pd.DataFrame, universe: list[str], **kwargs) -> None:
        """Estime le modèle de risque (si applicable)."""

    @abstractmethod
    def optimize(self, w_old: np.ndarray = None, is_first: bool = False) -> np.ndarray:
        """Produit les poids optimaux sous contraintes partagées."""

    def evaluate(self, w: np.ndarray, returns_oos: pd.DataFrame) -> dict:
        """Calcule toutes les métriques OOS (partagé)."""
```

---

### MOD-010 — bench_equal_weight

```python
class EqualWeight(BenchmarkModel):
    def fit(self, returns, universe, **kwargs):
        self.n = len(universe)

    def optimize(self, w_old=None, is_first=False):
        w = np.ones(self.n) / self.n
        # Hard cap à 5% non-binding pour n=1000
        return np.clip(w, 0, self.constraint_params['w_max'])
```

Pas de modèle de risque, pas de pénalité de turnover (turnover intrinsèquement faible).

---

### MOD-011 — bench_inverse_vol

```python
class InverseVolatility(BenchmarkModel):
    def fit(self, returns, universe, trailing_vol, current_date, **kwargs):
        """
        σ_i = trailing 252j annualized vol à current_date.
        w_i ∝ 1/σ_i, puis projection sur contraintes.
        """
        self.sigma = trailing_vol.loc[current_date, universe].values
        assert (self.sigma > 0).all(), "Zero or negative vol detected"

    def optimize(self, w_old=None, is_first=False):
        w = (1.0 / self.sigma)
        w /= w.sum()
        return self._project_to_constraints(w, w_old, is_first)

    def _project_to_constraints(self, w, w_old, is_first):
        """Clip w_max, zero < w_min, renormalize. Itérer 2-3 passes."""
```

---

### MOD-012 — bench_min_variance

```python
class MinimumVariance(BenchmarkModel):
    def fit(self, returns, universe, **kwargs):
        """
        Estime Σ_LW par Ledoit-Wolf (2004) shrinkage vers identité scalée
        sur la fenêtre d'entraînement complète (expanding, anti-cyclique).

        Utiliser sklearn.covariance.LedoitWolf ou implémentation directe.
        """
        R = returns[universe].dropna(how='all').values
        lw = LedoitWolf().fit(R)
        self.Sigma_LW = lw.covariance_

    def optimize(self, w_old=None, is_first=False):
        """
        min w^T Σ_LW w
        s.t. contraintes partagées (P_conc, P_turn, hard caps)

        QP convexe → CVXPY + MOSEK. Solution globale garantie.
        """
```

Exposer `self.Sigma_LW` pour réutilisation par MOD-013 (ERC).

---

### MOD-013 — bench_erc

```python
class EqualRiskContribution(BenchmarkModel):
    def fit(self, returns, universe, **kwargs):
        """Réutilise Σ_LW de MOD-012."""
        # Utiliser le même estimateur Ledoit-Wolf
        R = returns[universe].dropna(how='all').values
        lw = LedoitWolf().fit(R)
        self.Sigma_LW = lw.covariance_

    def optimize(self, w_old=None, is_first=False):
        """
        ERC via Spinu (2013) : formulation log-barrière convexe.

        min Σ_i Σ_j (w_i(Σw)_i - w_j(Σw)_j)²
        ou formulation équivalente :
        min (1/2) w^T Σ w - Σ_i ln(w_i)

        Convergence : < 5 itérations de Newton pour n < 1000.
        Projection post-hoc sur hard caps, puis renormalisation itérée.
        """
```

---

### MOD-014 — bench_pca_factor_rp

**C'est le benchmark le plus important** — il isole la valeur ajoutée du VAE vs la PCA linéaire.

```python
class PCAFactorRiskParity(BenchmarkModel):
    def fit(self, returns, universe, **kwargs):
        """
        1. PCA sur la matrice de rendements (T_est × n).
        2. Nombre de facteurs k par Bai & Ng (2002) IC₂.
        3. B_PCA ∈ R^(n × k) : loadings PCA.
        4. Σ_z_PCA = Λ_k (diagonale — composantes principales orthogonales).
        5. D_ε_PCA : résidus, variance diagonale, floor 1e-6.
        6. Σ_assets = B_PCA Λ_k B_PCA^T + D_ε_PCA.
        """

    def _bai_ng_ic2(self, returns_matrix, k_max=30):
        """
        Information Criterion IC₂ (Bai & Ng, 2002).
        IC₂(k) = ln(V(k)) + k · ((n+T)/(n·T)) · ln(min(n,T))
        V(k) = (1/(n·T)) · ||R - F_k Λ_k^T||²_F
        Sélectionner k* = argmin IC₂(k).
        Typiquement k ∈ [5, 15].
        """

    def optimize(self, w_old=None, is_first=False):
        """
        MÊME objectif et solver que le VAE :
        max -λ w^T Σ w + α H(w)  (μ = 0)

        H(w) calculé dans la base des facteurs principaux de Σ_z_PCA
        (mais puisque Σ_z_PCA = Λ_k est déjà diagonale, la rotation est triviale : V = I).

        SCA solver identique à MOD-008 (import sca_solver).
        Calibration α identique (frontière variance-entropie).
        Contraintes identiques.
        """
```

**Dépendances :** MOD-008 (C: import du SCA solver), MOD-001 (D: returns).

---

### MOD-015 — bench_pca_vol

```python
class PCAVolRiskParity(PCAFactorRiskParity):
    """
    Variante du MOD-014 avec matrice augmentée (T × 2n) :
    - Concaténation rendements z-scorés + volatilités réalisées 21j z-scorées.
    - PCA sur cette matrice augmentée.
    - Reste identique (IC₂, SCA, mêmes contraintes).

    Isolement : non-linéarité du VAE indépendamment de l'enrichissement features.
    """

    def fit(self, returns, universe, trailing_vol, **kwargs):
        # Z-score per window pour les two features, puis PCA sur matrice augmentée
```

---

### Tests requis (tous benchmarks)

1. `test_constraints_identical` : mêmes paramètres de contrainte pour tous les modèles
2. `test_equal_weight_sum_to_one` : w.sum() == 1
3. `test_min_var_beats_random` : min-var < random portfolio variance (sanity)
4. `test_erc_equal_risk_contributions` : RC_i ≈ RC_j ∀ i,j (tolérance 5%)
5. `test_pca_ic2_range` : k ∈ [3, 30] pour données réalistes
6. `test_pca_factor_rp_uses_sca` : le solver SCA converge
7. `test_benchmark_output_format` : tous les modèles retournent w de shape (n,)

---

## Sections ISD — Phase 4 (séquentiel)

---

### MOD-009 — walk_forward

**Phase :** 4 | **Mode :** lead_session | **Dépendances :** MOD-001–008 (D), MOD-010–015 (D) | **Densité :** haute
**Fichiers :** `src/walk_forward/*.py`, `tests/integration/test_walk_forward.py`

#### Objectif

Orchestrer la validation walk-forward complète : scheduling des ~34 folds, exécution de Phase A (HP selection) et Phase B (deployment) par fold, calcul des métriques sur 3 couches, scoring composite, et évaluation holdout finale.

#### Sous-tâche 1 : Scheduling des folds (folds.py)

```python
def generate_fold_schedule(
    total_years: int = 30,
    min_training_years: int = 10,
    oos_months: int = 6,
    embargo_days: int = 21,
    holdout_years: int = 3,
) -> list[dict]:
    """
    Génère le schedule des folds walk-forward.

    Fold k :
      - Training end : Year 10 + k × 0.5
      - Embargo : 21 trading days after training end
      - OOS start : training_end + embargo + 1
      - OOS end : OOS start + 6 months
      - Validation subset (nested) : [training_end - 2yr, training_end]
      - Training subset (nested) : [start, training_end - 2yr]

    Holdout : last ~3 years (Year 27 to Year 30)

    Returns: list of dicts with fold_id, train_start, train_end,
             val_start, val_end, embargo_start, embargo_end,
             oos_start, oos_end, is_holdout
    """
```

**~34 folds :** de Year 10 à Year 27, pas de 6 mois.

#### Sous-tâche 2 : Phase A — HP selection (phase_a.py)

```python
def run_phase_a(
    fold: dict,
    hp_configs: list[dict],  # grille d'hyperparamètres
    data_pipeline_output: dict,
    build_vae_fn,
) -> dict:
    """
    Pour chaque HP config :
      1. Construire le VAE avec ces HPs
      2. Entraîner sur [start, train_end - 2yr]
         avec early stopping sur [train_end - 2yr, train_end]
         → record E*_config
      3. Build downstream pipeline (AU, B_A, Σ_z, portfolio)
      4. Évaluer sur OOS → fold score

    Score composite :
      Score = Ĥ_OOS - λ_pen · max(0, MDD_OOS - MDD_threshold) - λ_est · max(0, 1 - R_Σ)

    où Ĥ = H(w) / ln(AU)  (entropie normalisée, ∈ [0, 1])
    MDD en fraction ∈ [0, 1]
    R_Σ = N_obs / (AU(AU+1)/2)

    Baselines : MDD_threshold = 0.20, λ_pen = 5, λ_est = 2.

    Élimination : configs avec AU < max(0.15K, AU_PCA), EP < max(0.40, EP_PCA),
    OOS/train MSE > 3.0 éliminées avant scoring.

    Returns: best_config, E_star, fold_scores
    """
```

#### Sous-tâche 3 : Phase B — Deployment run (phase_b.py)

```python
def run_phase_b(
    fold: dict,
    best_config: dict,
    E_star: int,           # epochs from Phase A
    data_pipeline_output: dict,
) -> dict:
    """
    Ré-entraîner l'encodeur sur TOUTES les données [start, train_end]
    pendant E* epochs (pas de validation set, pas d'early stopping).

    E* = median des E*_config across folds (robustesse).

    Sanity check : si training loss à E* en Phase B est > 20% inférieure
    à Phase A, flag le fold.

    Puis : pipeline downstream complet
    (AU, B_A, Σ_z, D_ε, portfolio optimization).

    Évaluer sur OOS.

    Returns: weights, metrics, AU, diagnostics
    """
```

#### Sous-tâche 4 : Métriques (metrics.py)

**Layer 1 — VAE quality :**
- OOS reconstruction error par régime (OOS/train MSE < 1.5 ; crisis/normal ∈ [0.5, 2.0])
- AU (AU ≥ max(0.15K, AU_PCA) ; AU ≤ min(0.85K, AU_max_stat))
- Latent stability : Spearman ρ > 0.85 des distances pairwise inter-stocks entre retrainings

**Layer 2 — Risk model quality :**
- Variance réalisée vs prédite : var(r_p^OOS) / (w^T Σ̂ w) ∈ [0.8, 1.2]
- Factor explanatory power : > max(0.50, EP_PCA + 0.10)
- Correlation réalisée vs prédite (rank)

**Layer 3 — Portfolio quality :**
- Entropie factorielle normalisée OOS (primary)
- Volatilité annualisée OOS (primary)
- Maximum drawdown OOS (primary)
- Rendement en période de crise (primary)
- Rendement annualisé, Sharpe, Calmar, Sortino (diagnostic)
- Turnover au rebalancement (diagnostic, cible < 30%)
- Diversification ratio DR (diagnostic)
- Nombre effectif de positions 1/Σw²_i (diagnostic)

#### Sous-tâche 5 : Holdout

```python
def run_holdout(
    final_config: dict,
    E_star_median: int,
    data_pipeline_output: dict,
    holdout_period: tuple,    # (start_date, end_date)
    benchmark_results: dict,  # résultats des benchmarks sur holdout
) -> dict:
    """
    UNE SEULE EXÉCUTION, à la fin.
    Entraîner chaque modèle (VAE + 6 benchmarks) sur tout l'historique
    jusqu'à t_holdout. Évaluer sur les ~3 dernières années.
    Comparer holdout vs walk-forward pour détecter surapprentissage structurel.
    """
```

#### Invariants applicables

- **INV-005 :** Aucun look-ahead. Training ≤ train_end, OOS > embargo_end.
- **CONV-09 :** Expanding window pour le training.
- **CONV-10 :** Point-in-time universe à chaque date.
- **INV-012 :** Mêmes contraintes pour VAE et benchmarks.

#### Tests requis

1. `test_fold_no_overlap` : aucun chevauchement training/OOS (avec embargo)
2. `test_fold_dates_sequential` : folds ordonnés chronologiquement
3. `test_holdout_untouched` : holdout data jamais vue pendant walk-forward
4. `test_score_normalization` : Ĥ ∈ [0, 1]
5. `test_phase_b_no_early_stopping` : E* epochs exécutés sans interruption

---

### MOD-016 — integration

**Phase :** 4 | **Mode :** lead_session | **Dépendances :** tous | **Densité :** haute
**Fichiers :** `src/integration/*.py`, `scripts/run_walk_forward.py`, `scripts/run_benchmarks.py`

#### Objectif

Orchestrer l'exécution complète end-to-end et produire le rapport de résultats avec tests statistiques de comparaison.

#### Sous-tâche 1 : Pipeline E2E (pipeline.py)

```python
class FullPipeline:
    def run(self, config: dict) -> dict:
        """
        1. data_pipeline.prepare(config)
        2. For each fold in walk_forward.schedule:
           a. Phase A: HP selection
           b. Phase B: Deployment run (VAE)
           c. For each benchmark: fit + optimize
        3. Aggregate results
        4. Run holdout
        5. Statistical tests
        6. Generate report
        """
```

#### Sous-tâche 2 : Tests statistiques (statistical_tests.py)

```python
def wilcoxon_paired_test(vae_scores, benchmark_scores):
    """
    Test de Wilcoxon signé (non-paramétrique) sur les différences par fold.
    H0 : médiane(Δ_k) = 0.
    Seuil : p < 0.05.
    """

def bootstrap_effect_size(vae_scores, benchmark_scores, n_bootstrap=10000):
    """Médiane de Δ + intervalle de confiance bootstrap (percentile method)."""

def holm_bonferroni_correction(p_values: list[float], alpha=0.05):
    """Correction pour 6 benchmarks × 4 métriques = 24 tests."""

def regime_decomposition(fold_metrics, vix_data):
    """
    Séparer folds en "crise" (> 20% jours VIX > P80) et "calme".
    Rapporter métriques et tests séparément.
    """
```

#### Tests requis

1. `test_pipeline_e2e_synthetic` : pipeline complet sur données synthétiques (50 stocks, 10 ans)
2. `test_statistical_tests_known` : Wilcoxon correct sur distributions connues
3. `test_holm_bonferroni` : correction appliquée correctement

---

## CLAUDE.md — Template pour le projet

```markdown
# Latent Risk Factor Discovery — VAE Strategy

## Context
Pipeline de construction de portefeuille basé sur la découverte de facteurs de risque latents
par VAE (1D CNN encoder-decoder). Objectif : maximiser la diversification factorielle
(entropie de Shannon sur les contributions au risque des facteurs principaux).

## Architecture
CRSP data → data_pipeline → VAE training → inference → risk_model → portfolio_optimization
                                                                         ↓
                                                                  walk_forward (34 folds)
                                                                         ↓
                                                                  benchmarks (×6)

## Conventions critiques — NE PAS VIOLER
- Rendements en LOG, jamais arithmétiques
- Z-score PER-WINDOW, PER-FEATURE
- σ² est un SCALAIRE, pas un vecteur (init 1.0, clamp [1e-4, 10])
- D = T × F DOIT apparaître dans la reconstruction loss (D/(2σ²) · L_recon)
- Modes P/F/A sont MUTUELLEMENT EXCLUSIFS
- Rescaling DUALE : date-specific pour estimation, current-date pour portefeuille
- Entropie H(w) calculée dans la base des FACTEURS PRINCIPAUX (après rotation V de Σ_z)
- Ratio σ_i,t / σ_bar_t WINSORISÉ [P5, P95] AVANT rescaling
- Contraintes de portefeuille IDENTIQUES entre VAE et tous les benchmarks
- AUCUN look-ahead — point-in-time strict

## Structure du code
- `src/data_pipeline/` : chargement, returns, univers, fenêtrage, crise
- `src/vae/` : architecture, loss, model
- `src/training/` : boucle d'entraînement, batching, early stopping
- `src/inference/` : profils composites, AU
- `src/risk_model/` : rescaling, régression factorielle, covariance
- `src/portfolio/` : entropie, SCA, contraintes, cardinalité
- `src/walk_forward/` : folds, Phase A/B, métriques
- `src/benchmarks/` : 6 modèles de benchmark
- `tests/unit/` et `tests/integration/`

## Workflow pour les agents
1. Lire `docs/isd/00_global.md` ET la section ISD de votre module
2. Implémenter les assertions d'interface EN PREMIER
3. TDD : test avant code pour chaque sous-tâche
4. Committer après chaque sous-tâche
5. Si ambigu : commentaire `# AMBIGUITY: ...` et interprétation conservatrice

## Dépendances
Python 3.11+, PyTorch ≥ 2.1, NumPy, SciPy, CVXPY + MOSEK, pandas, scikit-learn, pytest

## Tests
- `pytest tests/unit/` — tests unitaires par module
- `pytest tests/integration/` — tests d'intégration inter-modules
- Module complet ⟺ tous ses tests passent + assertions d'interface satisfaites
```

---

## Plan d'exécution opérationnel

### Phase 1 — Infrastructure (parallèle, Agent Team, ~2-3 jours)

| Teammate | Module | Livrable principal |
|----------|--------|-------------------|
| `data-engineer` | MOD-001 data_pipeline | Fenêtres z-scorées, univers, crisis labels, trailing vol |
| `ml-architect` | MOD-002 vae_architecture | VAEModel avec build_vae, sizing rules |
| `test-lead` | MOD-003 test_infrastructure | Données synthétiques, fixtures, solutions connues |

**Synchronisation :** assertions d'interface entre MOD-001 et MOD-002 (shape windows = (N, T, F), F=2, T=504).

### Phase 2 — Core pipeline (séquentiel, Subagents builder-validator, ~5-7 jours)

| Ordre | Module | Justification séquentielle |
|-------|--------|---------------------------|
| 1 | MOD-004 loss_function | Dépend de MOD-002 (VAEModel). Lead session (densité très haute). |
| 2 | MOD-005 training | Dépend de MOD-004 (loss). Lead session (couplage math degré 4). |
| 3 | MOD-006 inference | Dépend de MOD-005 (trained model). Subagent (densité moyenne). |
| 4 | MOD-007 risk_model | Dépend de MOD-006 (B matrix). Subagent (densité haute). |
| 5 | MOD-008 portfolio_optimization | Dépend de MOD-007 (Σ_z). Lead session (SCA solver très dense). |

**Protocole :** builder-validator pour MOD-006 et MOD-007. Lead session avec supervision humaine pour MOD-004, MOD-005, MOD-008.

### Phase 3 — Benchmarks (parallèle, Agent Team, ~2-3 jours)

| Teammate | Modules | Notes |
|----------|---------|-------|
| `bench-simple` | MOD-010 (1/N), MOD-011 (inverse-vol) | Trivial, ~1 jour |
| `bench-covariance` | MOD-012 (min-var), MOD-013 (ERC) | Partagent Σ_LW |
| `bench-factor` | MOD-014 (PCA factor RP), MOD-015 (PCA+vol) | Réutilisent SCA solver de MOD-008 |

**Dépendances :** MOD-014 et MOD-015 importent le SCA solver de MOD-008 — MOD-008 doit être complet et stable.

### Phase 4 — Intégration (séquentiel, lead session, ~3-5 jours)

| Ordre | Module | Description |
|-------|--------|-------------|
| 1 | MOD-009 walk_forward | Orchestration des 34 folds, Phase A/B |
| 2 | MOD-016 integration | E2E, tests statistiques, rapport final |

**Validation humaine obligatoire** avant passage Phase 3 → Phase 4 et avant exécution holdout.

---

## Matrice de décision post-exécution

| Scénario | Condition | Action |
|----------|-----------|--------|
| A — VAE surpasse tout | p < 0.05 sur ≥ 2/4 métriques primaires vs tous benchmarks | Production |
| B — VAE > PCA mais pas min-var/ERC | Non-linéarité utile, optimisation à revoir | Itérations 1-3 (doc v4.1 Section 6) |
| C — PCA ≈ VAE | Non-linéarité sans valeur mesurable | Adopter PCA (coût /100) |
| D — 1/N ≥ tous | Estimation error absorbe tout bénéfice | 1/N ou relâcher contraintes |
| E — Hétérogène par régime | VAE > en crise, < en calme | Système bi-régime |
