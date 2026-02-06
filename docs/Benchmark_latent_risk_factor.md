# Procédure de Benchmarking — Latent Risk Factor Discovery Strategy

## Objectif

Ce document décrit la procédure complète pour constituer un benchmark objectif contre lequel évaluer la stratégie VAE (document principal v4.1). Le principe directeur : la complexité du VAE n'est justifiée que si elle produit un bénéfice OOS mesurable et statistiquement significatif par rapport à des alternatives plus simples, sur les mêmes données, avec les mêmes contraintes, évaluées par les mêmes métriques.

---

## 1. Infrastructure partagée

Tous les modèles — VAE et benchmarks — doivent partager exactement la même infrastructure. Toute divergence invalide la comparaison.

### 1.1 Données

- **Source** : CRSP (ou alternative survivorship-bias-free : EODHD, Norgate Data).
- **Fréquence** : rendements logarithmiques journaliers, prix ajustés (dividendes + splits).
- **Historique** : 30 ans (T_année = 30), soit ~7 560 observations.
- **Traitement des valeurs manquantes** : forward-fill ≤ 5 jours consécutifs ; exclusion de la fenêtre au-delà.
- **Rendements de delisting** : convention Shumway (-30% NYSE/AMEX, -55% Nasdaq).

### 1.2 Univers d'investissement

Identique pour tous les modèles, reconstitué point-in-time à chaque date de rebalancement :

| Critère | Seuil |
|---------|-------|
| Capitalisation flottante | Entrée ≥ $500M, sortie ≥ $400M |
| Volume quotidien moyen (ADV) | ≥ $2M (trailing 3 mois) |
| Historique de cotation | ≥ 504 jours (~2 ans) |
| Type | Actions ordinaires (pas d'ETF, ADR, REIT, SPAC, warrants) |
| Marché | NYSE + NASDAQ + AMEX |
| Sélection si excédent | Top $n$ = 1 000 par capitalisation flottante |

L'univers d'entraînement (pour les modèles qui en ont besoin) inclut toutes les actions ayant figuré dans l'univers d'investissement à n'importe quelle date du fold d'entraînement, y compris les actions délistées depuis.

### 1.3 Contraintes de portefeuille

Appliquées de manière identique à tous les modèles pour isoler l'effet du modèle de risque :

- **Long-only** : $w_i \geq 0$
- **Fully invested** : $\sum w_i = 1$
- **Position max** : $w_{\max}^{\text{hard}} = 5\%$
- **Position min** : $w_i = 0$ ou $w_i \geq 0.10\%$
- **Soft threshold concentration** : $\bar{w} = 3\%$, pénalité $P_{\text{conc}}(w) = \sum_i \max(0, w_i - \bar{w})^2$
- **Turnover** : pénalité Almgren-Chriss (linéaire $\kappa_1$ + quadratique $\kappa_2$) avec hard cap $\tau_{\max}^{\text{hard}} = 30\%$ one-way
- **Premier rebalancement** : $\kappa_1 = \kappa_2 = 0$

Les paramètres de pénalité ($\phi$, $\kappa_1$, $\kappa_2$, $\bar{\delta}$) sont identiques pour tous les modèles. Ils sont fixés une seule fois (via le walk-forward de la stratégie VAE) puis appliqués tels quels aux benchmarks — ne pas les ré-optimiser par modèle, sinon chaque benchmark bénéficie de son propre tuning et la comparaison mesure la qualité du tuning, pas du modèle de risque.

### 1.4 Protocole walk-forward

Strictement identique au protocole de la stratégie principale (Section 4.8 du document v4.1) :

- **Fenêtre d'entraînement** : expansive (tout l'historique disponible jusqu'à $t_k$).
- **Embargo** : 21 jours.
- **Période OOS** : ~6 mois par fold.
- **Minimum d'entraînement** : ~10 ans.
- **Holdout final** : dernières ~3 années, réservées, intouchables jusqu'à l'évaluation finale.
- **Nombre de folds** : ~34 (identiques pour tous les modèles — mêmes dates de coupure).

Chaque modèle produit un portefeuille au début de chaque fold OOS, le maintient fixe pendant 6 mois, et les métriques sont calculées sur cette période. Le portefeuille est reconstitué au début du fold suivant.

### 1.5 Fréquence de rebalancement

Alignée sur le VAE : mensuelle à trimestrielle (utiliser la même cadence pour tous les modèles). Entre les rebalancement, les poids dérivent passivement avec les prix.

---

## 2. Modèles de benchmark

Six benchmarks ordonnés par complexité croissante. Chacun isole une composante spécifique de la chaîne de valeur de la stratégie VAE.

### 2.1 Equal-weight (1/N)

**Principe** : $w_i = 1/n$ pour les $n$ actions de l'univers.

**Ce que ça teste** : si aucune optimisation, aucun modèle de risque, et aucune estimation de paramètre ne peut faire mieux que la diversification naïve, toute complexité supplémentaire est inutile. DeMiguel, Garlappi & Uppal (2009) ont montré que 1/N bat la plupart des méthodes sophistiquées avec des fenêtres d'estimation typiques — l'erreur d'estimation absorbe le bénéfice de l'optimisation.

**Implémentation** :
- Input : liste des $n$ actions de l'univers à la date de rebalancement.
- Modèle de risque : aucun.
- Optimisation : aucune ($w_i = 1/n$).
- Contraintes : seul le hard cap à 5% s'applique (binding si $n < 20$, non-binding pour $n = 1000$). Pas de pénalité de turnover (le turnover est intrinsèquement faible — uniquement l'entrée/sortie de l'univers).

**Composante VAE testée** : la chaîne complète. Si le VAE ne bat pas 1/N, rien ne fonctionne.

### 2.2 Minimum-variance (Ledoit-Wolf)

**Principe** : $\min_w \; w^T \hat{\Sigma}_{\text{LW}} w$ sous les contraintes partagées.

**Ce que ça teste** : si une estimation directe de la covariance (sans modèle factoriel) suffit à produire un portefeuille de risque minimal, le modèle factoriel VAE n'apporte rien. Ce benchmark est le test décisif pour la composante "estimation de covariance factorielle" du pipeline.

**Implémentation** :
- Input : matrice de rendements journaliers $(T_{\text{estimation}} \times n)$ sur la fenêtre d'entraînement du fold.
- Modèle de risque : covariance échantillon shrinkée Ledoit-Wolf (2004) vers identité scalée. Estimée sur la fenêtre d'entraînement complète (anti-cyclique, cohérent avec le VAE). Alternative : Ledoit-Wolf non-linéaire (2020) pour test de sensibilité.
- Optimisation : QP convexe (CVXPY + MOSEK). Le problème est standard $\min w^T \Sigma w$ s.t. contraintes — solution globale garantie, aucune multi-start nécessaire.
- Contraintes : identiques au VAE ($P_{\text{conc}}$, $P_{\text{turn}}$, hard caps).

**Composante VAE testée** : la valeur ajoutée de la décomposition factorielle $\Sigma = B\Sigma_z B^T + D_\varepsilon$ vs l'estimation directe $\hat{\Sigma}_{\text{LW}}$.

### 2.3 Risk parity classique — Equal Risk Contribution (ERC)

**Principe** : trouver $w$ tel que chaque action contribue également au risque total du portefeuille. La contribution au risque de l'action $i$ est $\text{RC}_i = w_i \cdot (\Sigma w)_i$, et ERC impose $\text{RC}_i = \text{RC}_j \;\forall\; i,j$.

**Ce que ça teste** : si l'idée de "diversification maximale" appliquée au niveau des actifs (pas des facteurs) produit des résultats comparables, la décomposition factorielle n'apporte pas de valeur. ERC partage la philosophie du VAE (aucun actif ne domine le risque) mais opère en dimension asset-space, pas factor-space.

**Implémentation** :
- Input : matrice de covariance $\hat{\Sigma}_{\text{LW}}$ (même estimateur que le benchmark 2.2).
- Modèle de risque : Ledoit-Wolf, identique au benchmark 2.2.
- Optimisation : formulation log-barrière convexe de Spinu (2013) — convergence en < 5 itérations de Newton pour $n < 1000$, solution globale garantie. Projection sur les contraintes de hard cap post-hoc (re-normalisation, itérée si nécessaire).
- Contraintes : identiques. Les pénalités de turnover sont appliquées de la même manière.

**Composante VAE testée** : la valeur ajoutée de la diversification factorielle (entropie sur les facteurs) vs la diversification asset-level (ERC sur les actifs).

### 2.4 Inverse-volatilité

**Principe** : $w_i \propto 1/\sigma_i$ où $\sigma_i$ est la volatilité trailing 252 jours de l'action $i$. Aucune estimation de corrélation nécessaire.

**Ce que ça teste** : si ignorer entièrement les corrélations et allouer simplement en inverse de la volatilité individuelle suffit, toute la machinerie d'estimation de covariance (factorielle ou directe) est superflue. Ce benchmark a un seul paramètre (la fenêtre de volatilité) et zéro erreur d'estimation sur les corrélations — il ne peut pas overfitter les corrélations puisqu'il ne les utilise pas.

**Implémentation** :
- Input : volatilité trailing 252 jours de chaque action.
- Modèle de risque : aucun modèle de corrélation (les positions sont déterminées par les volatilités individuelles uniquement).
- Optimisation : calcul direct $w_i = (1/\sigma_i) / \sum_j (1/\sigma_j)$, puis projection sur les contraintes (hard cap, min position).
- Contraintes : identiques.

**Composante VAE testée** : la valeur ajoutée de l'estimation de corrélation (factorielle ou non) vs l'allocation par volatilité pure.

### 2.5 PCA factor risk parity

**Principe** : extraire $k$ facteurs par PCA sur la matrice de rendements, estimer la covariance factorielle, puis maximiser l'entropie de Shannon des contributions au risque dans l'espace factoriel principal — exactement le même objectif que la stratégie VAE mais avec des facteurs linéaires.

**Ce que ça teste** : c'est le benchmark le plus important. Il isole précisément la valeur ajoutée du VAE en tant qu'extracteur de facteurs non-linéaire. Si la PCA produit les mêmes résultats, les non-linéarités capturées par le VAE ne contribuent pas à la diversification, et l'encodeur neuronal est un coût inutile.

**Implémentation** :
- Input : matrice de rendements journaliers $(T_{\text{estimation}} \times n)$ sur la fenêtre d'entraînement.
- Extraction factorielle : PCA standard. Nombre de facteurs $k$ sélectionné par Bai & Ng (2002) IC₂ sur la même matrice — typiquement $k \in [5, 15]$. Le critère est recalculé à chaque fold.
- Matrice d'exposition : $B_{\text{PCA}} \in \mathbb{R}^{n \times k}$ (les loadings PCA).
- Covariance factorielle : $\Sigma_z^{\text{PCA}} = \Lambda_k$ (diagonale — les composantes principales sont orthogonales par construction). Pas de shrinkage nécessaire puisque $\Sigma_z$ est déjà diagonale.
- Idiosyncratic : $D_\varepsilon^{\text{PCA}}$ estimé sur les résidus.
- Covariance asset : $\Sigma_{\text{assets}} = B_{\text{PCA}} \Lambda_k B_{\text{PCA}}^T + D_\varepsilon^{\text{PCA}}$.
- Optimisation : entropie de Shannon sur les contributions factorielles, résolue par le même SCA solver que le VAE (multi-start, Armijo, cardinality enforcement). Formulation default identique : $\max_w -\lambda w^T \Sigma w + \alpha H(w)$ avec $\mu = 0$.
- Contraintes : identiques au VAE.
- Calibration $\alpha$ : même procédure (frontière variance-entropie, sélection du coude) avec $\lambda = 1.0$.

**Composante VAE testée** : la valeur ajoutée de l'extracteur non-linéaire (VAE) vs l'extracteur linéaire (PCA) — auto-pruning vs IC₂, K ≈ 80 facteurs vs k ≈ 10, non-linéarités vs linéarité.

### 2.6 PCA factor risk parity + realized volatility feature

**Principe** : variante du benchmark 2.5 qui intègre la volatilité réalisée comme feature supplémentaire. La PCA est appliquée non pas sur la matrice de rendements seule, mais sur une matrice augmentée $(T \times 2n)$ concaténant les rendements z-scorés et les volatilités réalisées 21 jours z-scorées de chaque action. Alternativement : PCA séparée sur rendements et sur volatilités, puis concaténation des loadings.

**Ce que ça teste** : si le bénéfice du VAE vient principalement de l'input enrichi (F = 2) plutôt que de la non-linéarité de l'encodeur, ce benchmark le révélera. Il isole l'effet "données plus riches" de l'effet "modèle plus complexe".

**Implémentation** :
- Input : matrice augmentée rendements + volatilité réalisée, z-scorées par fenêtre.
- Le reste est identique au benchmark 2.5 (PCA, IC₂, SCA, même objectif, mêmes contraintes).

**Composante VAE testée** : la non-linéarité de l'encodeur indépendamment de l'enrichissement des features.

---

## 3. Métriques de comparaison

Trois couches, toutes calculées out-of-sample sur les folds walk-forward. Les métriques sont calculées par fold, puis agrégées (médiane + intervalle interquartile sur les ~34 folds).

### 3.1 Métriques primaires — diversification et risque

Ce sont les métriques qui déterminent la décision. Elles correspondent directement à l'objectif déclaré de la stratégie : minimiser le risque par diversification factorielle.

| Métrique | Définition | Cible | Justification |
|----------|------------|-------|---------------|
| **Entropie factorielle normalisée OOS** ($\hat{H}_{\text{OOS}}$) | $H(w) / \ln(\text{AU})$ pour le VAE ; $H(w) / \ln(k)$ pour PCA ; non applicable pour 1/N et inverse-vol | Maximiser | Objectif propre de la stratégie — mesure la qualité de la diversification factorielle, normalisée par la capacité disponible |
| **Volatilité annualisée OOS** | $\sigma_p = \text{std}(r_p) \times \sqrt{252}$ sur la période OOS | Minimiser | Mesure directe du risque total |
| **Maximum drawdown OOS** | Plus grande perte cumulée peak-to-trough sur la période OOS | Minimiser | Capture le risque de queue — l'événement qui détruit le capital |
| **Rendement en période de crise** | Rendement cumulé du portefeuille sur les jours où VIX > 80ᵉ percentile (expanding window, point-in-time) | Minimiser la perte | Test direct de la protection anti-cyclique — la raison d'être de la stratégie |

### 3.2 Métriques diagnostiques — performance et efficience

Ces métriques ne déterminent pas la décision mais contextualisent les résultats. Un modèle qui diversifie parfaitement mais détruit de la valeur de manière systématique a un problème.

| Métrique | Définition | Interprétation |
|----------|------------|----------------|
| **Rendement annualisé OOS** | Rendement log moyen $\times$ 252 | Sanity check — la diversification ne doit pas détruire de valeur |
| **Sharpe ratio OOS** | Rendement excédentaire (vs taux sans risque) / volatilité | Performance ajustée au risque — diagnostic, pas objectif primaire |
| **Calmar ratio** | Rendement annualisé / max drawdown | Rendement par unité de risque de queue |
| **Sortino ratio** | Rendement excédentaire / downside deviation | Pénalise uniquement la volatilité baissière |
| **Turnover au rebalancement** | $\frac{1}{2} \sum_i |w_i^{\text{new}} - w_i^{\text{old}}|$ | Stabilité du modèle de risque (cible < 30%) |
| **Diversification ratio (DR)** | $\sum_i w_i \sigma_i / \sigma_p$ (Choueifaty & Coignard 2008) | Complément asset-level à l'entropie factorielle |
| **Nombre effectif de positions** | $1 / \sum_i w_i^2$ (inverse HHI) | Concentration asset-level |

### 3.3 Métriques de qualité du modèle de risque

Applicable uniquement aux modèles qui estiment une covariance ($\Sigma$) — pas au 1/N ni à l'inverse-volatilité.

| Métrique | Définition | Cible |
|----------|------------|-------|
| **Variance réalisée vs prédite** | $\text{var}(r_p^{\text{OOS}}) / (w^T \hat{\Sigma} w)$ | Ratio ∈ [0.8, 1.2] — la covariance estimée prédit correctement le risque réalisé |
| **Pouvoir explicatif factoriel** | $1 - \text{tr}(D_\varepsilon) / \text{tr}(\Sigma_{\text{assets}})$ | > 0.50 (la structure factorielle explique plus de la moitié de la variance) |
| **Corrélation réalisée vs prédite** | Corrélation rang entre les $\binom{n}{2}$ corrélations prédites par $\hat{\Sigma}$ et les corrélations réalisées OOS | Indicateur de la qualité structurelle de $\hat{\Sigma}$ |

---

## 4. Tests statistiques

Un modèle qui semble "meilleur" sur la médiane peut ne pas l'être de manière significative. Les tests suivants quantifient la confiance.

### 4.1 Test apparié par fold

Pour chaque paire (VAE, benchmark $b$), calculer la différence de score par fold :

$$\Delta_k = \text{Score}_{\text{VAE}}^{(k)} - \text{Score}_b^{(k)}, \qquad k = 1, \ldots, K_{\text{folds}}$$

Le score est le score composite de la Section 4.8 du document principal : $\hat{H}_{\text{OOS}} - \lambda_{\text{pen}} \cdot \max(0, \text{MDD} - \text{MDD}_{\text{threshold}})$.

- **Test de Wilcoxon signé** (non-paramétrique, pas d'hypothèse de normalité) sur $\{\Delta_k\}$. $H_0$ : médiane de $\Delta = 0$. Seuil : $p < 0.05$.
- **Taille d'effet** : médiane de $\Delta$ et intervalle de confiance bootstrap (10 000 réplications, percentile method).

### 4.2 Test par métrique individuelle

Pour chaque métrique primaire séparément (volatilité, MDD, rendement crise), appliquer le même test de Wilcoxon apparié. Cela identifie sur quelle dimension le VAE gagne ou perd.

### 4.3 Correction pour tests multiples

Avec 6 benchmarks × 4 métriques primaires = 24 tests, appliquer la correction de Holm-Bonferroni pour contrôler le taux d'erreur global (FWER) à 5%.

### 4.4 Robustesse par régime

Séparer les folds en deux groupes : "crise" (folds contenant > 20% de jours avec VIX > 80ᵉ percentile) et "calme" (le reste). Rapporter les métriques et tests séparément pour chaque groupe. La stratégie VAE est conçue pour surperformer en crise — si elle ne le fait que sur les folds calmes, l'hypothèse fondamentale est invalidée.

---

## 5. Procédure d'exécution — étapes séquentielles

### Étape 1 — Préparation des données (commune)

1. Télécharger les données CRSP (ou alternative) couvrant 30 ans + 21 jours de warm-up.
2. Calculer les rendements logarithmiques journaliers avec prix ajustés.
3. Appliquer les corrections Shumway pour les rendements de delisting.
4. Reconstituer l'univers point-in-time à chaque date de rebalancement.
5. Définir les $K_{\text{folds}}$ dates de coupure du walk-forward (identiques pour tous les modèles).
6. Pré-calculer les volatilités trailing 252 jours par action et les percentiles VIX (expanding window) pour la classification crise/calme.

### Étape 2 — Implémentation des benchmarks (2–4 semaines)

Ordre recommandé (du plus simple au plus complexe) :

1. **1/N** : implémentation triviale (quelques lignes), sert de sanity check pour l'infrastructure (walk-forward, calcul des métriques, gestion de l'univers).
2. **Inverse-volatilité** : ajoute le calcul de volatilité trailing, pas d'estimation de covariance.
3. **Minimum-variance** : première implémentation nécessitant l'estimation $\hat{\Sigma}_{\text{LW}}$ et l'optimisation QP. Valider que le solver (CVXPY + MOSEK) fonctionne avec les contraintes partagées.
4. **ERC** : réutilise $\hat{\Sigma}_{\text{LW}}$ du min-var. Ajoute le solver Spinu.
5. **PCA factor risk parity** : ajoute l'extraction PCA, le calcul de $B_{\text{PCA}}$, la décomposition covariance factorielle, et le SCA solver pour l'entropie. Partage le solver SCA avec le VAE.
6. **PCA + vol réalisée** : variante du 5 avec matrice augmentée.

### Étape 3 — Exécution du walk-forward (par modèle)

Pour chaque fold $k = 1, \ldots, K_{\text{folds}}$ et chaque modèle :

1. Reconstruire l'univers $\mathcal{U}_{t_k}$.
2. Extraire la matrice de rendements de la fenêtre d'entraînement $[\text{start}, t_k]$.
3. Estimer le modèle de risque (selon le benchmark).
4. Optimiser le portefeuille sous contraintes partagées.
5. Simuler les rendements du portefeuille sur la période OOS ($t_k + 21\text{j}$ à $t_{k+1}$), poids fixes, drift passif.
6. Calculer toutes les métriques sur la période OOS.

### Étape 4 — Exécution de la stratégie VAE

Identique à l'étape 3 mais avec le pipeline complet décrit dans le document v4.1 (Sections 4.2–4.7). Utiliser les mêmes dates de coupure, le même univers, les mêmes contraintes, les mêmes métriques.

### Étape 5 — Agrégation et tests statistiques

1. Compiler la matrice de résultats : $(K_{\text{folds}} \times 7 \text{ modèles} \times M \text{ métriques})$.
2. Calculer la médiane et l'IQR de chaque métrique par modèle.
3. Exécuter les tests de Wilcoxon appariés (VAE vs chaque benchmark) par métrique.
4. Appliquer la correction de Holm-Bonferroni.
5. Séparer les résultats crise/calme et répéter les tests.

### Étape 6 — Évaluation holdout (une seule fois, à la fin)

Après sélection de la configuration finale via les étapes 1–5 :

1. Entraîner chaque modèle sur toutes les données jusqu'à $t_{\text{holdout}}$.
2. Évaluer sur les ~3 dernières années réservées.
3. Comparer les résultats holdout aux résultats walk-forward pour détecter un éventuel surapprentissage structurel (la configuration sélectionnée exploite-t-elle une spécificité des folds de sélection ?).

---

## 6. Matrice de décision

Les résultats des benchmarks déterminent la suite du projet selon la grille suivante :

### Scénario A — Le VAE surpasse tous les benchmarks

Le VAE bat statistiquement ($p < 0.05$, Wilcoxon apparié) les 6 benchmarks sur au moins 2 métriques primaires sur 4, sans dégradation significative sur les autres. La complexité est justifiée. Passer à l'implémentation production (Phase B du walk-forward).

### Scénario B — Le VAE bat PCA factor risk parity mais pas min-var/ERC

La décomposition factorielle non-linéaire apporte de la valeur vs la linéaire, mais l'optimisation par entropie factorielle ne surpasse pas l'optimisation directe en espace asset. Investiguer : le problème est-il dans l'entropie factorielle ou dans l'estimation de Σ_z ? Appliquer les itérations 1–3 de la Section 8 du document principal.

### Scénario C — PCA factor risk parity ≈ VAE

Les non-linéarités du VAE n'apportent rien de mesurable. La PCA capture l'essentiel de la structure factorielle. Adopter la PCA (Itération 6 du document principal) — même pipeline d'optimisation, facteurs linéaires, coût computationnel divisé par ~100.

### Scénario D — 1/N ≈ ou > tous les modèles

Aucun modèle d'optimisation ne justifie sa complexité (résultat cohérent avec DeMiguel et al. 2009 pour un univers equity-only de grande taille). Deux options : (a) adopter 1/N et réallouer l'effort de recherche, ou (b) investiguer si les contraintes de portefeuille trop serrées (hard cap 5%, turnover 30%) compriment les différences entre modèles. Relâcher les contraintes et re-tester.

### Scénario E — Les résultats sont hétérogènes par régime

Le VAE surpasse en crise mais sous-performe en calme (ou inversement). C'est le scénario le plus informatif : il valide partiellement l'hypothèse anti-cyclique. Option : système à deux régimes (VAE en crise détectée, PCA/ERC en calme) ou application des itérations 2–3 du document principal (blending Σ_z + mode directionnel).

---

## 7. Synthèse — matrice modèles × composantes testées

| Benchmark | Modèle de risque | Objectif d'optimisation | Composante VAE isolée |
|-----------|-----------------|------------------------|----------------------|
| 1/N | Aucun | Aucun | Chaîne complète |
| Inverse-volatilité | Volatilité individuelle (pas de corrélation) | Allocation proportionnelle | Valeur de l'estimation de corrélation |
| Min-variance (LW) | Covariance directe shrinkée | $\min w^T\Sigma w$ | Valeur de la décomposition factorielle |
| ERC | Covariance directe shrinkée | Égalisation des contributions au risque asset-level | Valeur de la diversification factorielle vs asset-level |
| PCA factor risk parity | PCA linéaire + Σ_z diagonale | $\max -\lambda w^T\Sigma w + \alpha H(w)$ | Valeur de la non-linéarité du VAE |
| PCA + vol réalisée | PCA linéaire sur features enrichis | $\max -\lambda w^T\Sigma w + \alpha H(w)$ | Non-linéarité indépendamment de l'enrichissement des données |

Chaque benchmark neutralise un étage de complexité du VAE. Si le VAE ne surpasse pas un benchmark, l'étage correspondant ne contribue pas.

---

## Références

- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy? *Review of Financial Studies*, 22(5), 1915–1953.
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- Ledoit, O. & Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics*, 48(5), 3043–3065.
- Spinu, F. (2013). An Algorithm for Computing Risk Parity Weights. *SSRN Working Paper* 2297383.
- Bai, J. & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models. *Econometrica*, 70(1), 191–221.
- Clarke, R., de Silva, H. & Thorley, S. (2013). Risk Parity, Maximum Diversification, and Minimum Variance: An Analytic Perspective. *Journal of Portfolio Management*, 39(3), 39–53.
- Choueifaty, Y. & Coignard, Y. (2008). Toward Maximum Diversification. *Journal of Portfolio Management*, 35(1), 40–51.
- Feng, Y. & Palomar, D. P. (2015). SCRIP: Successive Convex Optimization Methods for Risk Parity Portfolio Design. *IEEE Transactions on Signal Processing*, 63(19), 5285–5300.
- Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions. *Journal of Risk*, 3(2), 5–39.
