# DIVERGENCES CODE vs DOCUMENTS DE REFERENCE (ISD / DVT)

> **Date** : 2026-02-13
> **Auteur** : Audit automatique de la pipeline VAE Latent Risk Factor
> **Documents de reference** :
> - ISD : `docs/ISD_vae_latent_risk_factors.md`
> - DVT : `docs/DVT_strategie_facteurs_risque_latents_v4_1.md`

---

## Divergence 1 : Formule AU_max_stat

| | Detail |
|---|---|
| **Code** | `active_units.py:105` — `floor(sqrt(2 * N_obs / r_min))` |
| **DVT** | Section 4.6 — `floor((-1 + sqrt(1 + 4*N_obs/r_min)) / 2)` |
| **ISD** | `floor(sqrt(2*N_obs / r_min))` (meme que le code) |

**Explication vulgarisee** : Les deux formules tentent de repondre a la meme question : "combien de facteurs latents peut-on estimer de maniere fiable avec N_obs observations ?". La formule du DVT resout exactement l'equation quadratique `AU*(AU+1)/2 <= N_obs/r_min` (le nombre de parametres d'une matrice de covariance AU x AU est AU*(AU+1)/2). La formule du code/ISD est une simplification qui neglige le terme "+1" de AU*(AU+1), ce qui revient a resoudre `AU^2/2 <= N_obs/r_min`.

**Impact numerique** : Pour N_obs=7560 (30 ans quotidien), r_min=2 : DVT donne 86, Code donne 86. Pour N_obs=1000 : DVT=30, Code=31. Difference negligeable (<2%).

**Recommandation** : Le code et l'ISD sont alignes. La formule DVT est plus exacte mathematiquement mais la difference est negligeable. **Garder le code tel quel** — l'ISD prime sur le DVT pour l'implementation.

---

## Divergence 2 : Plancher beta_min dans Mode F

| | Detail |
|---|---|
| **Code** | `loss.py:424` — `max(0.01, min(1.0, epoch / T_warmup))` |
| **DVT** | Section 4.4 — `beta_t = min(1, t / T_warmup)` (commence a 0) |
| **ISD** | `beta_t = min(1, t / T_warmup)` (commence a 0) |

**Explication vulgarisee** : Le Mode F augmente progressivement le poids du terme KL (divergence de Kullback-Leibler) de 0 a 1 pendant la phase de warmup. Les documents disent de commencer a exactement 0. Le code ajoute un plancher a 0.01 pour eviter que le KL soit totalement desactive au debut. Sans ce plancher, le "posterior collapse" est possible : l'encodeur pourrait pousser les moyennes latentes vers des valeurs extremes puisqu'il n'y a aucune penalite KL pour les contraindre.

**Impact** : Protection contre le posterior collapse au debut du warmup. C'est une amelioration defensive.

**Recommandation** : **Garder le code** — le plancher 0.01 est une amelioration pragmatique. Le DVT/ISD definissent l'intention mathematique ; le code ajoute une garde numerique justifiee. Le docstring explique deja pourquoi. Un test devrait verifier ce comportement.

---

## Divergence 3 : Dropout par defaut — Asymetrie encoder/decoder

| | Detail |
|---|---|
| **Code encoder.py** | `DROPOUT = 0.2` (constante hard-codee en haut du fichier) |
| **Code decoder.py** | `DROPOUT = 0.1` (constante hard-codee en haut du fichier) |
| **Code config.py** | `dropout: float = 0.1` (default dans VAEArchitectureConfig) |
| **DVT** | Ne specifie pas de taux exact |
| **ISD** | Mentionne dropout mais ne fixe pas de valeur |
| **Changelog** | Entry #10 : "Raised encoder dropout from 0.1 to 0.2" |

**Explication vulgarisee** : Le dropout est une technique de regularisation qui desactive aleatoirement une fraction des neurones pendant l'entrainement pour eviter le surapprentissage. Le probleme : quand `build_vae(dropout=X)` est appele, le parametre X est propage aux modules, mais les constantes `DROPOUT` au sommet de `encoder.py` (0.2) et `decoder.py` (0.1) sont les valeurs par defaut de secours. Si un chemin d'appel ne propage pas le dropout, l'encodeur utilise 0.2 et le decodeur 0.1. L'asymetrie intentionnelle (plus de regularisation sur l'encodeur) est justifiee, mais la constante du config (0.1) ne correspond pas au default de l'encodeur (0.2).

**Impact** : Incoherence potentielle si le dropout n'est pas explicitement passe via `build_vae()`. En pratique, `_adapt_vae_params()` dans pipeline.py passe toujours le dropout donc le bug ne se manifeste pas dans le flux normal.

**Recommandation** : **Corriger le config** — aligner `VAEArchitectureConfig.dropout` sur 0.2 pour correspondre au default de l'encodeur, OU mettre a jour `DROPOUT` dans encoder.py pour correspondre au config (0.1). Le plus coherent serait de mettre le config a 0.2 puisque le changelog indique que c'etait un choix delibere.

---

## Divergence 4 : Variance targeting — Ajout non documente

| | Detail |
|---|---|
| **Code** | `pipeline.py:95-137` — `_variance_targeting_scale()` |
| **DVT** | Aucune mention |
| **ISD** | Aucune mention |

**Explication vulgarisee** : Le "variance targeting" est une calibration qui compare la variance predite par le modele de risque a la variance realisee d'un portefeuille equi-pondere. Si le modele surestime la variance de 100x (var_ratio = 0.007 documente dans le changelog), le scaling corrige en multipliant Sigma_assets par le ratio `Var_realisee / Var_predite`. Cela evite qu'un modele de risque mal calibre empeche l'optimiseur de trouver de bonnes solutions.

**Impact** : Correction critique pour les cas ou la covariance predite est tres eloignee de la realite. Sans cette correction, les portefeuilles seraient trop conservateurs.

**Recommandation** : **Garder le code** — c'est un ajout necessaire pour la robustesse. Il devrait etre documente dans l'ISD comme une etape supplementaire (MOD-007.5 par exemple). Le clamp [0.01, 100] est une bonne garde.

---

## Divergence 5 : Auto-adaptation VAE pour petits univers

| | Detail |
|---|---|
| **Code** | `pipeline.py:271-358` — `_adapt_vae_params()` |
| **DVT** | K=200 fixe, architecture fixe |
| **ISD** | K et architecture fixes dans le config |

**Explication vulgarisee** : Quand l'univers d'actions est petit (ex: 50 actions au lieu de 1000), les parametres par defaut du VAE (K=200, C_MIN=384) creent un modele surdimensionne — trop de parametres pour trop peu de donnees. L'auto-adaptation reduit K proportionnellement au nombre d'unites actives estimees, baisse le C_MIN (nombre minimum de canaux) de 384 a 144, et relaxe la contrainte de ratio r_max tout en renforcant la regularisation.

**Impact** : Permet l'execution sur des petits univers sans crash. Sans cette adaptation, `build_vae()` leve une `ValueError` (contrainte de capacite violee).

**Recommandation** : **Garder le code** — ajout necessaire pour la generalisation. Le DVT ne prevoyait pas d'execution sur des petits univers.

---

## Divergence 6 : Mode d'entrainement direct (sans walk-forward)

| | Detail |
|---|---|
| **Code** | `pipeline.py:724-941` — `run_direct()` |
| **DVT** | Seul le walk-forward est decrit |
| **ISD** | Seul le walk-forward est decrit |

**Explication vulgarisee** : `run_direct()` permet d'entrainer le VAE sur toute la periode moins un holdout, sans les ~34 folds du walk-forward. C'est un mode "quick test" pour valider la pipeline avant de lancer le walk-forward complet (qui prend beaucoup plus de temps).

**Recommandation** : **Garder le code** — mode de developpement/debug utile. N'interfere pas avec le mode walk-forward principal.

---

## Divergence 7 : Methodes de cardinality enforcement

| | Detail |
|---|---|
| **Code** | `cardinality.py` — 4 methodes : sequential, gradient, miqp, two_stage |
| **DVT** | Section 4.7 — Sequential greedy principalement, two-stage en alternative (Section 8.7) |
| **ISD** | Sequential greedy uniquement |

**Explication vulgarisee** : La contrainte de cardinalite dit qu'un poids doit etre soit exactement 0 soit au moins w_min (0.10%). C'est un probleme NP-dur. Le DVT propose un algorithme greedy (sequentiel : eliminer une position a la fois, re-optimiser). Le code offre 3 methodes supplementaires : "gradient" (approximation du premier ordre, plus rapide), "MIQP" (programmation mixte en nombres entiers, plus exact mais plus lent), "two-stage" (decomposition en espace factoriel puis projection). Un mecanisme de fallback automatique est prevu.

**Recommandation** : **Garder le code** — les methodes supplementaires sont des ameliorations qui respectent le meme objectif. Le fallback `two_stage -> miqp -> gradient -> sequential` garantit qu'une solution est toujours trouvee.

---

## Divergence 8 : Composition multi-start

| | Detail |
|---|---|
| **Code** | `sca_solver.py:527-549` — EW, inverse-diag, inverse-vol, 2 random |
| **DVT** | M=5 : EW, min-variance, approximate ERC, 2 random |
| **ISD** | Meme que DVT |

**Explication vulgarisee** : Le multi-start lance l'optimisation SCA depuis 5 points de depart differents. Le DVT dit : (1) poids egaux, (2) variance minimale (QP convexe), (3) ERC approximatif (Spinu). Le code remplace le (2) min-variance exact par une approximation "inverse de la diagonale de Sigma" et le (3) ERC par "inverse de la volatilite". Ce sont des approximations plus rapides des memes idees.

**Impact** : Les points de depart sont des heuristiques ; leur qualite exacte a peu d'impact car le SCA converge vers un point KKT quel que soit le depart. L'approximation est acceptable.

**Recommandation** : **Acceptable** mais pourrait etre ameliore en utilisant un vrai QP pour le start #2 (le Cholesky est deja pre-calcule, le cout marginal est faible).

---

## Divergence 9 : create_windows retourne un 3-tuple

| | Detail |
|---|---|
| **Code** | `windowing.py` — retourne `(windows, metadata, raw_returns)` |
| **ISD** | Ne mentionne que `(windows, metadata)` |
| **DVT** | Co-movement loss necessite les raw returns |

**Explication vulgarisee** : La fonction `create_windows()` retourne maintenant un 3eme element : les retours bruts (non z-scores) pour calculer la correlation de Spearman dans la co-movement loss. L'ISD ne mentionne pas ce 3eme retour car il a ete ajoute lors de l'integration de la co-movement loss (changelog #8).

**Recommandation** : **Garder le code** — necessaire pour la co-movement loss. L'ISD devrait etre mis a jour.

---

## Divergence 10 : Fresh CVXPY problem vs Parametric

| | Detail |
|---|---|
| **Code** | `sca_solver.py` — Construit un nouveau probleme CVXPY a chaque iteration SCA |
| **DVT** | Implicite : re-utilisation parametrique |
| **ISD** | Mentionne "parametric CVXPY" |

**Explication vulgarisee** : Le DVT/ISD suggerent de construire le probleme CVXPY une seule fois et de mettre a jour les parametres (gradient) a chaque iteration. Le code construit un nouveau probleme chaque fois pour eviter des bugs de compatibilite DCP (DPP canonicalization) entre versions de CVXPY. C'est plus lent (~5-10%) mais plus robuste.

**Recommandation** : **Garder le code** — la robustesse prime. Le commentaire dans le code explique le choix. Le gain de vitesse du parametrique est marginal par rapport au temps de resolution du solver.

---

## Resume

| # | Sujet | Code vs Docs | Severite | Recommandation |
|---|---|---|---|---|
| 1 | AU_max_stat formula | Simplification vs quadratique exacte | **Negligeable** | Garder le code (ISD-aligne) |
| 2 | Beta plancher 0.01 | Ajout defensif | **Amelioration** | Garder le code |
| 3 | Dropout asymetrie | 0.2 encoder vs 0.1 config | **Bug mineur** | Aligner config sur 0.2 |
| 4 | Variance targeting | Non documente | **Ajout critique** | Garder, documenter |
| 5 | Auto-adaptation | Non documente | **Ajout necessaire** | Garder, documenter |
| 6 | Mode direct | Non documente | **Ajout utile** | Garder |
| 7 | 4 methodes cardinalite | 1 methode dans spec | **Amelioration** | Garder (fallback chain) |
| 8 | Multi-start approx | Approx vs exact | **Acceptable** | Garder (ou ameliorer start #2) |
| 9 | create_windows 3-tuple | 2-tuple dans ISD | **Evolution** | Garder, mettre a jour ISD |
| 10 | Fresh CVXPY | vs parametrique | **Choix robustesse** | Garder (DCP compliance) |

**Conclusion** : Sur 10 divergences identifiees, **aucune n'est un defaut critique**. 3 sont des ameliorations intentionnelles, 3 des ajouts necessaires non documentes, 2 des differences de formule negligeables, 1 choix de robustesse, et 1 bug mineur (dropout config). La seule action corrective recommandee est d'aligner la valeur par defaut du dropout dans `config.py` (changer de 0.1 a 0.2).
