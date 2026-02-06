# Implementation Specification Document (ISD) — Méthodologie Générale

## Pour l'orchestration multi-agents de projets à couplage technique dense

**Version 1.0**

---

## Table des matières

- [Introduction](#introduction)
- [Partie A — Utilisateur : substrat irréductible](#partie-a--utilisateur--substrat-irréductible)
  - [A.1 Prérequis environnementaux](#a1-prérequis-environnementaux)
  - [A.2 Production du Document de Vision Technique (DVT)](#a2-production-du-document-de-vision-technique-dvt)
  - [A.3 Identification des invariants critiques domaine-spécifiques](#a3-identification-des-invariants-critiques-domaine-spécifiques)
  - [A.4 Définition des critères d'acceptation globaux](#a4-définition-des-critères-dacceptation-globaux)
  - [A.5 Livrables de la Partie A — Checklist](#a5-livrables-de-la-partie-a--checklist)
- [Partie B — Agent IA supervisé : compilation structurelle](#partie-b--agent-ia-supervisé--compilation-structurelle)
  - [B.1 Analyse du DVT et extraction de la topologie](#b1-analyse-du-dvt-et-extraction-de-la-topologie)
  - [B.2 Décomposition modulaire](#b2-décomposition-modulaire)
  - [B.3 Construction du graphe de dépendances](#b3-construction-du-graphe-de-dépendances)
  - [B.4 Classification séquentiel / parallèle](#b4-classification-séquentiel--parallèle)
  - [B.5 Estimation de la densité de contexte par module](#b5-estimation-de-la-densité-de-contexte-par-module)
  - [B.6 Production des assertions inter-modules](#b6-production-des-assertions-inter-modules)
  - [B.7 Génération de l'ISD structuré](#b7-génération-de-lisd-structuré)
  - [B.8 Validation humaine et itération](#b8-validation-humaine-et-itération)
  - [B.9 Livrables de la Partie B — Checklist](#b9-livrables-de-la-partie-b--checklist)
- [Partie C — Agents autonomes : exécution orchestrée](#partie-c--agents-autonomes--exécution-orchestrée)
  - [C.1 Architecture d'orchestration](#c1-architecture-dorchestration)
  - [C.2 Configuration du CLAUDE.md projet](#c2-configuration-du-claudemd-projet)
  - [C.3 Structure d'une section ISD consommable par un agent](#c3-structure-dune-section-isd-consommable-par-un-agent)
  - [C.4 Protocole builder-validator](#c4-protocole-builder-validator)
  - [C.5 Gestion des phases d'exécution](#c5-gestion-des-phases-dexécution)
  - [C.6 Conventions de communication inter-agents](#c6-conventions-de-communication-inter-agents)
  - [C.7 Critères de complétion et gestion d'erreur](#c7-critères-de-complétion-et-gestion-derreur)
  - [C.8 Templates opérationnels](#c8-templates-opérationnels)
  - [C.9 Anti-patterns et diagnostics](#c9-anti-patterns-et-diagnostics)
  - [C.10 Livrables de la Partie C — Checklist](#c10-livrables-de-la-partie-c--checklist)
- [Annexe — Glossaire](#annexe--glossaire)

---

## Introduction

### Problème adressé

Les projets à couplage technique dense — pipelines de recherche quantitative, systèmes de modélisation scientifique, architectures ML end-to-end — présentent une caractéristique structurelle qui les rend résistants aux approches standard d'orchestration multi-agents : les décisions de conception à un point du pipeline ont des ramifications en cascade, souvent non-linéaires, sur des composants distants. Une modification de la fonction de loss affecte l'auto-pruning, qui conditionne le nombre de facteurs actifs, qui contraint l'estimation de la covariance en aval, qui détermine le comportement du solver de portefeuille.

Les approches existantes — Ralph (boucle autonome sur PRD JSON), agent teams natifs de Claude Code (parallélisation par défaut), subagents (délégation focalisée) — fournissent des primitives d'orchestration puissantes mais aucune ne résout le problème de **compilation** : transformer un document méthodologique dense en spécifications auto-suffisantes, correctement segmentées, et annotées pour l'orchestration.

### Ce que ce document formalise

Ce document définit une méthode en trois phases pour produire un **Implementation Specification Document (ISD)** — un artefact intermédiaire entre le document de vision technique (rédigé par l'humain) et le code (produit par les agents). L'ISD est le produit d'une analyse croisée entre la structure de dépendance du domaine et les contraintes opérationnelles des agents (fenêtre de contexte, absence de mémoire inter-sessions, communication limitée).

### Périmètre et limites

Cette méthodologie s'applique aux projets satisfaisant trois conditions : (1) complexité technique suffisante pour justifier une orchestration multi-agents (> 2 000 lignes de code attendues, > 5 modules interdépendants), (2) existence d'un document méthodologique ou de spécification détaillé, (3) couplage inter-modules qui empêche une décomposition naïve en tâches indépendantes.

Elle ne s'applique pas aux projets CRUD, aux refactorings de modules isolés, ni aux prototypes exploratoires où l'itération rapide en session unique est plus efficace.

---

## Partie A — Utilisateur : substrat irréductible

Cette partie décrit ce que l'humain doit produire **avant toute intervention d'un agent IA**. Ces éléments constituent le substrat irréductible du projet : la connaissance domaine-spécifique, les choix architecturaux fondamentaux, et la configuration environnementale que seul l'expert du domaine peut fournir.

### A.1 Prérequis environnementaux

#### A.1.1 Installation et configuration de Claude Code

L'environnement d'exécution doit être préparé avant la production de l'ISD. Les étapes suivantes sont impératives et non-déléguables à un agent.

**Claude Code.** S'assurer que Claude Code est installé et fonctionnel. Vérifier la version (`claude --version`) et la disponibilité des outils nécessaires (bash, read, write, edit, glob, grep). Configurer le modèle par défaut selon les besoins de complexité du projet.

**Agent Teams (expérimental).** Si le projet requiert de la parallélisation inter-agents (voir B.4 pour les critères de décision), activer la fonctionnalité :

```json
// settings.json ou variable d'environnement
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

Note : Agent Teams est expérimental. Les limitations connues concernent la reprise de session, la coordination des tâches, et le comportement de shutdown. Pour les projets critiques, prévoir des checkpoints manuels.

**Terminal.** Le mode split-pane (visualisation simultanée des teammates) requiert tmux ou iTerm2. Le mode in-process (par défaut) fonctionne dans tout terminal. Le terminal intégré de VS Code, Windows Terminal, et Ghostty ne supportent pas le split-pane.

**Git.** Initialiser le dépôt du projet avant le début de l'orchestration. Les agents utiliseront Git comme mécanisme de persistence et de rollback. Configurer les branches de travail selon le pattern prévu (une branche par phase, par module, ou par agent — à décider en B.4).

#### A.1.2 Structure du projet

Créer la structure minimale de répertoires :

```
project_root/
├── .claude/
│   ├── settings.json          # Configuration Claude Code
│   ├── agents/                # Définitions des subagents (Part C)
│   └── commands/              # Commandes slash personnalisées
├── CLAUDE.md                  # Contexte projet global (Part C)
├── docs/
│   ├── dvt.md                 # Document de Vision Technique (Part A)
│   ├── isd/                   # Sections de l'ISD (Part B)
│   │   ├── 00_global.md       # Invariants et conventions globales
│   │   ├── 01_module_xxx.md   # Une section par module
│   │   └── ...
│   └── assertions/            # Assertions inter-modules (Part B)
│       └── contracts.yaml     # Contrats d'interface formalisés
├── src/                       # Code produit par les agents
│   ├── module_xxx/
│   └── ...
├── tests/                     # Tests unitaires et d'intégration
│   ├── unit/
│   └── integration/
└── scripts/                   # Scripts utilitaires
```

#### A.1.3 Gestion des dépendances

L'utilisateur doit spécifier l'environnement d'exécution cible : version de Python (ou autre langage), dépendances système, bibliothèques requises et leurs versions. Cette information est critique car les agents ne doivent pas prendre de décisions de dépendances autonomes — un choix de bibliothèque (e.g., PyTorch vs JAX, CVXPY vs SciPy) a des ramifications architecturales que seul l'expert domaine peut arbitrer.

Formaliser dans un fichier de spécification (e.g., `requirements.txt`, `pyproject.toml`, `environment.yml`) **avant** le début de l'orchestration.

### A.2 Production du Document de Vision Technique (DVT)

Le DVT est le document source à partir duquel l'ISD sera compilé. C'est le seul document que l'utilisateur doit rédiger intégralement. Sa qualité détermine directement la qualité de l'ISD et, par conséquent, la qualité du code produit par les agents.

#### A.2.1 Contenu requis

Le DVT doit contenir, au minimum :

**Vision et objectif.** L'intention du projet formulée en termes vérifiables. Non pas "construire un bon modèle" mais "produire un modèle dont la métrique X sur les données Y dépasse le seuil Z". L'objectif doit être suffisamment précis pour qu'un agent puisse déterminer si le projet est terminé.

**Architecture globale.** Une description du pipeline end-to-end, incluant le flux de données entre composants. Un diagramme (même textuel) est indispensable. L'architecture doit spécifier les points d'entrée (données sources), les transformations successives, et les sorties (livrables).

**Spécifications techniques par composant.** Pour chaque composant du pipeline :
- Les entrées attendues (types, shapes, conventions de nommage, unités).
- Les transformations appliquées (formules mathématiques complètes, algorithmes, pseudo-code).
- Les sorties produites (mêmes spécifications que les entrées).
- Les paramètres configurables (valeurs par défaut, plages valides, justification).
- Les cas limites et leur traitement.

**Choix de conception et justifications.** Chaque décision non-triviale doit être accompagnée de sa justification. Cette information est critique pour les agents : elle leur permet de distinguer ce qui est modifiable (choix d'implémentation) de ce qui est contraint (choix de conception intentionnel). Un agent qui ne dispose pas de cette information risque de "corriger" un choix délibéré en croyant résoudre un bug.

**Protocole de validation.** Comment le projet sera évalué : métriques, protocole expérimental (e.g., walk-forward), benchmarks, critères de succès/échec. Ce protocole doit être suffisamment détaillé pour être implémentable sans ambiguïté.

**Références.** Tout article, documentation, ou source externe dont les agents pourraient avoir besoin. Les agents n'ont pas nécessairement accès à ces sources — les éléments pertinents doivent être cités et résumés dans le DVT.

#### A.2.2 Niveau de détail requis

Le niveau de détail du DVT conditionne l'autonomie des agents. La règle directrice :

> **Tout élément qui, s'il est mal implémenté, invalide silencieusement le pipeline en aval, doit être spécifié à un niveau de détail qui ne laisse aucune place à l'interprétation.**

Cela signifie concrètement :
- Les formules mathématiques doivent être écrites en notation explicite, avec les dimensions des tenseurs et les conventions d'indexation.
- Les constantes doivent avoir des valeurs numériques, pas des descriptions qualitatives ("un petit nombre" → $10^{-8}$).
- Les cas limites doivent être traités explicitement ("si le dénominateur est < $10^{-8}$, clamper à $10^{-8}$").
- Les interactions entre composants doivent être documentées ("la modification de X affecte Y via le mécanisme Z").

#### A.2.3 Indicateurs de complétude

Le DVT est suffisamment complet lorsque :

1. **Test du lecteur naïf expert.** Un ingénieur compétent dans le domaine technique (mais ignorant du projet) pourrait implémenter chaque composant sans poser de questions de clarification.
2. **Test de l'implémentation alternative.** Deux implémentations indépendantes produites à partir du DVT donneraient des résultats numériquement identiques (aux erreurs de floating-point près).
3. **Test de la suppression.** Aucun paragraphe ne peut être supprimé sans qu'un agent ne puisse produire une implémentation incorrecte.

### A.3 Identification des invariants critiques domaine-spécifiques

Les invariants critiques sont des propriétés du système qui doivent être vraies à tout moment et dont la violation n'est pas détectable par des tests unitaires standards. Ils résultent de la connaissance domaine-spécifique de l'utilisateur et ne peuvent pas être dérivés automatiquement du DVT.

#### A.3.1 Catégories d'invariants

**Invariants mathématiques.** Propriétés algébriques qui doivent être respectées dans le code. Exemples : une matrice de covariance doit être semi-définie positive ; un facteur de normalisation doit être présent pour que deux termes d'une loss soient comparables ; une convention de signe doit être cohérente entre composants.

**Invariants de cohérence inter-modules.** Propriétés qui portent sur l'interface entre deux modules et qui ne sont vérifiables que par inspection conjointe. Exemple : le tenseur de sortie du module A a une shape $(n, K)$ qui doit correspondre à l'entrée attendue par le module B, mais $K$ est déterminé dynamiquement par le module A (auto-pruning) et le module B doit s'adapter.

**Invariants de convention.** Choix de convention qui, s'ils diffèrent entre modules, produisent des résultats numériquement faux sans erreur explicite. Exemples : rendements en log vs arithmétiques ; covariance par observation vs par élément ; indices 0-based vs 1-based.

**Invariants de sécurité financière.** Spécifiques aux projets quantitatifs : absence de look-ahead (aucune donnée future dans le calcul), point-in-time universe construction, traitement correct des delistings, gestion des dividendes et splits.

#### A.3.2 Formalisation

Chaque invariant doit être formalisé selon le template suivant :

```yaml
invariant:
  id: INV-001
  category: mathematical | inter_module | convention | safety
  description: "Le facteur D = T × F doit apparaître dans le numérateur
                de la reconstruction loss pour équilibrer avec le KL"
  modules_affected: [loss_function, training, monitoring]
  violation_consequence: "Posterior collapse — toutes les dimensions latentes
                         convergent vers le prior, le modèle devient inutile"
  detection_method: "Vérifier que AU > 0.15 × K après entraînement.
                     Si AU < 5, le facteur D est probablement manquant."
  test_assertion: |
    # La loss reconstruction doit être multipliée par D/(2*sigma_sq)
    # et non par 1/(2*sigma_sq)
    assert loss_recon_coeff == (T * F) / (2 * sigma_sq)
```

#### A.3.3 Processus d'extraction

L'identification des invariants est un exercice délibéré, pas un sous-produit de la rédaction du DVT. L'utilisateur doit, pour chaque composant du pipeline :

1. Lister les hypothèses implicites que le composant fait sur ses entrées.
2. Lister les propriétés que le composant garantit sur ses sorties.
3. Identifier les propriétés qui traversent les frontières de modules (c'est-à-dire les hypothèses d'un module qui dépendent des garanties d'un autre).
4. Pour chaque propriété inter-module, se demander : "Si un agent implémente correctement le module A et un autre agent implémente correctement le module B, mais avec des conventions différentes, le pipeline produit-il silencieusement des résultats faux ?" Si oui, c'est un invariant critique.

### A.4 Définition des critères d'acceptation globaux

Les critères d'acceptation définissent les conditions sous lesquelles le projet est considéré comme terminé. Ils diffèrent des métriques de performance en ce qu'ils incluent des critères structurels et de qualité de code, pas seulement des résultats numériques.

#### A.4.1 Critères structurels

- **Couverture de test.** Seuil minimal de couverture (e.g., > 80% des lignes, > 90% des branches critiques).
- **Reproductibilité.** Étant donné un seed fixe, deux exécutions du pipeline complet produisent des résultats identiques.
- **Documentation.** Chaque fonction publique a une docstring spécifiant entrées, sorties, et comportement aux cas limites.
- **Conformité aux invariants.** Tous les invariants de A.3 sont vérifiés par des tests automatisés.

#### A.4.2 Critères fonctionnels

- **Complétude.** Tous les composants du DVT sont implémentés et intégrés.
- **Exécution end-to-end.** Le pipeline complet s'exécute sans erreur sur les données de test.
- **Benchmarks.** Les métriques de performance atteignent les seuils définis dans le DVT.

#### A.4.3 Critères de qualité de code

- **Absence de code mort.** Pas de fonctions inutilisées, de branches inaccessibles, ou de commentaires TODO non-résolus.
- **Modularité.** Chaque module est importable et testable indépendamment.
- **Performance.** Les contraintes de temps d'exécution spécifiées dans le DVT sont respectées.

### A.5 Livrables de la Partie A — Checklist

À l'issue de la Partie A, l'utilisateur doit avoir produit :

- [ ] Environnement Claude Code configuré et fonctionnel
- [ ] Agent Teams activé (si parallélisation requise)
- [ ] Structure de répertoires créée
- [ ] Dépendances spécifiées (`requirements.txt` ou équivalent)
- [ ] DVT complet (docs/dvt.md) satisfaisant les trois tests de complétude
- [ ] Invariants critiques formalisés (docs/assertions/invariants.yaml)
- [ ] Critères d'acceptation globaux définis
- [ ] Dépôt Git initialisé avec un premier commit contenant tous les livrables

---

## Partie B — Agent IA supervisé : compilation structurelle

Cette partie décrit le travail d'analyse que l'agent IA effectue sous la supervision directe de l'utilisateur. L'objectif est de **compiler** le DVT en un ISD structuré — c'est-à-dire transformer un document méthodologique dense et couplé en spécifications auto-suffisantes, correctement segmentées, et annotées pour l'orchestration multi-agents.

Chaque étape de cette partie produit un artefact intermédiaire soumis à la validation de l'utilisateur avant de passer à l'étape suivante. L'agent ne doit pas procéder à l'étape $n+1$ sans validation explicite de l'étape $n$.

### B.1 Analyse du DVT et extraction de la topologie

#### B.1.1 Objectif

Identifier la **structure logique** du projet telle qu'elle émerge du DVT : les composants fonctionnels, leurs relations, les flux de données, et les points de couplage. Cette analyse est distincte de la décomposition modulaire (B.2) — elle porte sur la topologie du problème, pas sur la segmentation du code.

#### B.1.2 Procédure

L'agent effectue une lecture analytique du DVT et produit :

**Inventaire des composants fonctionnels.** Chaque composant est une unité logique qui transforme des entrées en sorties. Il ne correspond pas nécessairement à un module de code — un composant peut être réparti sur plusieurs modules, ou plusieurs composants peuvent cohabiter dans un module.

**Graphe de flux de données.** Un DAG (Directed Acyclic Graph) où les nœuds sont les composants et les arêtes représentent les flux de données. Chaque arête est annotée avec le type de données transmises (shape du tenseur, type, convention).

**Matrice de couplage.** Pour chaque paire de composants $(i, j)$, estimer le degré de couplage sur une échelle :
- **0 — Indépendant.** Aucune interaction. Les composants peuvent être implémentés sans connaissance l'un de l'autre.
- **1 — Interface.** Couplage uniquement par l'interface d'entrée/sortie. Spécifier le contrat d'interface suffit.
- **2 — Paramétrique.** Un composant produit un paramètre qui affecte le comportement (pas seulement les données) de l'autre. Exemple : le nombre de dimensions actives (AU) déterminé par le VAE conditionne la taille des matrices dans le risk model.
- **3 — Sémantique.** Les composants partagent des conventions implicites dont la violation produit des résultats numériquement faux sans erreur. Exemple : la convention de normalisation (per-element vs per-window) doit être identique entre la loss function et le monitoring.
- **4 — Mathématique.** Les composants sont liés par une identité mathématique dont la violation invalide la preuve de convergence, la justification théorique, ou le comportement attendu du système. Exemple : le facteur $D$ dans la loss relie reconstruction et KL d'une manière qui garantit le balancement auto-régulé.

**Points de couplage critiques.** Les paires avec un degré de couplage ≥ 3 sont des points de couplage critiques. L'agent les liste avec une description de l'invariant qui les relie. Ces points détermineront les contraintes de séquentialité en B.4.

#### B.1.3 Format de sortie

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
      invariant: "INV-001 — facteur D dans la loss"
      description: "La loss function et le monitoring doivent utiliser
                    la même convention D = T × F pour que σ² soit interprétable"
```

#### B.1.4 Point de validation utilisateur

L'utilisateur vérifie :
- Tous les composants du DVT sont représentés.
- Les degrés de couplage sont corrects (l'agent tend à sous-estimer le couplage sémantique et mathématique — l'expert domaine est le seul juge fiable).
- Les points de couplage critiques n'ont pas d'omissions.

### B.2 Décomposition modulaire

#### B.2.1 Objectif

Transformer la topologie fonctionnelle (B.1) en une décomposition en **modules de code** — des unités implémentables, testables unitairement, et assignables à un agent.

#### B.2.2 Principes de décomposition

La décomposition modulaire optimise simultanément trois critères en tension :

**Minimisation du couplage inter-modules.** Les frontières de modules doivent passer par les arêtes de couplage minimal (degré 0 ou 1) du graphe topologique. Deux composants avec un couplage de degré ≥ 3 ne doivent **jamais** être séparés dans des modules différents, sauf si le couplage peut être entièrement capturé par un contrat d'interface formalisé.

**Maximisation de la cohésion intra-module.** Un module doit réaliser une fonction logiquement complète. Le test : un développeur peut comprendre le module sans lire les autres, en se basant uniquement sur les contrats d'interface.

**Respect de la contrainte de contexte.** Chaque module doit pouvoir être spécifié (formules, pseudo-code, tests) dans un volume inférieur à ~60% de la fenêtre de contexte d'un agent (réservant ~40% pour le code produit et les interactions). Pour Claude Code avec Opus 4.6 (1M tokens), cela représente ~600K tokens de spécification — généralement non-binding. Pour des modèles plus petits ou des contextes réduits, cette contrainte devient active et force une granularité plus fine.

#### B.2.3 Procédure

1. **Partitionnement initial.** Regrouper les composants fonctionnels (B.1) en clusters de couplage ≥ 2, en utilisant le graphe de couplage. Chaque cluster devient un module candidat.

2. **Vérification de cohérence.** Pour chaque module candidat, vérifier qu'il réalise une fonction compréhensible isolément. Si un module contient des composants fonctionnellement disjoints (regroupés uniquement par couplage), envisager de scinder en préservant les interfaces de couplage.

3. **Vérification de taille.** Estimer le volume de spécification de chaque module (voir B.5). Si un module dépasse la contrainte de contexte, le scinder en sous-modules — mais uniquement le long de frontières de couplage ≤ 2. Si aucune telle frontière n'existe, le module est **irréductible** et doit être implémenté par un agent en session unique avec contexte complet.

4. **Ordonnancement.** Définir un ordre d'implémentation qui respecte les dépendances : un module ne peut être implémenté qu'après ceux dont il dépend (degré de couplage ≥ 2 entrant).

#### B.2.4 Format de sortie

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
    irréductible: true
    reason: "Couplage mathématique degré 4 entre les trois composants"
```

### B.3 Construction du graphe de dépendances

#### B.3.1 Objectif

Formaliser le DAG des dépendances inter-modules sous une forme directement transposable en task list pour l'orchestration (Claude Code Agent Teams ou subagents).

#### B.3.2 Types de dépendances

**Dépendance de données (D).** Le module B consomme une sortie du module A. Le module B ne peut pas être testé sans un stub ou un mock de la sortie de A.

**Dépendance d'interface (I).** Le module B doit connaître l'interface de sortie de A (types, shapes) pour définir son interface d'entrée, mais ne consomme pas les données réelles de A pendant le développement. Le développement peut être parallélisé si l'interface est gelée.

**Dépendance de code (C).** Le module B importe et utilise du code du module A (fonctions utilitaires, classes partagées). Le module A doit être implémenté et stable avant B.

**Dépendance de validation (V).** Le module B ne peut être validé que conjointement avec A (tests d'intégration). L'implémentation peut être parallèle, mais la validation est séquentielle.

#### B.3.3 Formalisation du DAG

```yaml
dependency_graph:
  edges:
    - from: MOD-001  # data_pipeline
      to: MOD-003    # loss_function
      type: D        # dépendance de données
      interface: "windows: Tensor(n × T × F)"
      blockant: true
      
    - from: MOD-002  # vae_architecture
      to: MOD-003    # loss_function
      type: C        # dépendance de code
      interface: "VAEModel class with forward() method"
      blockant: true
      
    - from: MOD-001  # data_pipeline
      to: MOD-002    # vae_architecture
      type: I        # dépendance d'interface seulement
      interface: "input_shape: (T, F)"
      blockant: false  # parallélisable si interface gelée
      
  critical_path:
    - [MOD-001, MOD-003, MOD-004, MOD-005, MOD-006, MOD-007]
    
  parallelizable_groups:
    - [MOD-001, MOD-002, MOD-008]  # aucune dépendance mutuelle
    - [MOD-009, MOD-010, MOD-011]  # benchmarks indépendants
```

#### B.3.4 Point de validation utilisateur

L'utilisateur vérifie que :
- Le chemin critique est correct (aucune dépendance manquante qui créerait un blocage en cours d'exécution).
- Les dépendances de type I sont réellement non-bloquantes (l'interface est suffisamment stable pour être gelée).
- Les groupes parallélisables sont effectivement indépendants.

### B.4 Classification séquentiel / parallèle

#### B.4.1 Objectif

Pour chaque module, déterminer le mode d'exécution optimal : session lead, subagent, ou teammate d'un agent team. Cette classification est le produit de l'analyse croisée entre le graphe de dépendances (B.3) et les caractéristiques opérationnelles de chaque primitif d'orchestration.

#### B.4.2 Matrice de décision

| Critère | Session lead | Subagent | Agent Team teammate |
|---------|-------------|----------|---------------------|
| Couplage avec autres modules | Degré ≥ 3 avec le module en cours | Degré ≤ 2, séquentiel | Degré ≤ 1, parallèle |
| Densité de contexte | Très haute (> 70% fenêtre) | Moyenne à haute | Faible à moyenne |
| Besoin de validation humaine en cours | Oui (choix de conception) | Non (spécification complète) | Non |
| Communication inter-agents requise | N/A | Non (rapport au lead) | Oui (partage de findings) |
| Durée estimée | Variable | < 1h de contexte agent | < 1h de contexte agent |
| Indépendance | N/A | Haute (après dépendances résolues) | Très haute |
| Coût en tokens | 1x | 1x par subagent | 1x par teammate (additif) |

#### B.4.3 Heuristiques de classification

**Session lead** — utiliser pour :
- Les modules irréductibles (B.2.3) dont la spécification dépasse la capacité d'un subagent.
- Les modules nécessitant des décisions de conception en temps réel (itération avec l'utilisateur).
- L'intégration finale et les tests end-to-end.

**Subagent** — utiliser pour :
- Les modules bien spécifiés avec des dépendances résolues (sorties des modules précédents disponibles).
- Les tâches de refactoring, documentation, ou écriture de tests après l'implémentation principale.
- Le pattern builder-validator (un subagent code, un autre valide — voir C.4).

**Agent Team teammate** — utiliser pour :
- Les groupes de modules parallélisables identifiés en B.3.
- Les modules indépendants de taille comparable (éviter le déséquilibre où un teammate finit bien avant les autres).
- Les phases d'exploration (comparer deux approches d'implémentation en parallèle, puis sélectionner la meilleure).

#### B.4.4 Annotation des modules

Chaque module reçoit une annotation d'orchestration :

```yaml
modules:
  - id: MOD-003
    name: "loss_function"
    orchestration:
      mode: "lead_session"
      justification: "Couplage mathématique degré 4 ; densité de contexte > 70% ;
                      interaction avec 3 modes (P/F/A) nécessitant compréhension globale"
      phase: 2
      
  - id: MOD-001
    name: "data_pipeline"
    orchestration:
      mode: "agent_team_teammate"
      justification: "Indépendant ; couplage interface seulement ;
                      parallélisable avec MOD-002 et MOD-008"
      phase: 1
      team: "infrastructure"
```

#### B.4.5 Définition des phases d'exécution

Les modules sont regroupés en **phases** séquentielles, chaque phase contenant des modules exécutables en parallèle :

```
Phase 1 (parallèle) : MOD-001, MOD-002, MOD-008
    ↓ synchronisation
Phase 2 (séquentiel) : MOD-003 → MOD-004 → MOD-005 → MOD-006 → MOD-007
    ↓ synchronisation
Phase 3 (parallèle) : MOD-009, MOD-010, MOD-011, MOD-012
    ↓ synchronisation
Phase 4 (intégration) : MOD-013
```

### B.5 Estimation de la densité de contexte par module

#### B.5.1 Objectif

Estimer, pour chaque module, le volume de spécification que l'agent devra maintenir en contexte pendant l'implémentation. Cette estimation détermine si un module peut être assigné à un subagent (contexte borné) ou nécessite une session lead (contexte complet).

#### B.5.2 Méthode d'estimation

Pour chaque module, comptabiliser :

**Spécifications mathématiques.** Nombre de formules, de cas de bord, de paramètres. Pondérer par la complexité : une formule simple (1 ligne) = 1 unité ; une formule avec des cas conditionnels, des clamping, des interactions = 3–5 unités.

**Interfaces d'entrée/sortie.** Nombre de types de données distincts en entrée et sortie, avec leurs conventions (shapes, unités, normalisations).

**Invariants applicables.** Nombre d'invariants (A.3) qui portent sur ce module. Chaque invariant doit être en contexte pendant l'implémentation.

**Code existant à connaître.** Si le module dépend de code de modules précédents (dépendance C), les interfaces de ce code doivent être en contexte.

**Estimation volumétrique.** En première approximation :

$$V_{\text{contexte}}(\text{module}) \approx V_{\text{spec}} + V_{\text{interfaces}} + V_{\text{invariants}} + V_{\text{code\_dep}}$$

Où chaque $V$ est estimé en tokens (~0.75 mots par token en anglais technique, ~4 caractères par token en code Python).

#### B.5.3 Seuils

| Densité | Volume estimé | Mode recommandé |
|---------|--------------|-----------------|
| Faible | < 10K tokens | Subagent ou teammate |
| Moyenne | 10K – 50K tokens | Subagent préférable |
| Haute | 50K – 200K tokens | Session lead ou subagent avec contexte étendu |
| Très haute | > 200K tokens | Session lead obligatoire ; envisager la scission du module |

### B.6 Production des assertions inter-modules

#### B.6.1 Objectif

Produire un ensemble d'assertions formelles qui servent de **contrat** entre modules. Ces assertions sont le mécanisme principal de détection d'incohérence lorsque des modules sont implémentés par des agents différents.

#### B.6.2 Types d'assertions

**Assertions de shape.** Vérification des dimensions des tenseurs aux interfaces.

```python
# Assertion : la sortie de l'encodeur a la bonne shape
assert encoder_output.shape == (n_stocks, K), \
    f"Encoder output shape {encoder_output.shape} != expected ({n_stocks}, {K})"
```

**Assertions de type et de plage.** Vérification que les valeurs sont dans les plages attendues.

```python
# Assertion : σ² est scalaire, positif, et dans la plage de clamp
assert sigma_sq.ndim == 0, "σ² must be scalar"
assert 1e-4 <= sigma_sq.item() <= 10, f"σ² = {sigma_sq.item()} outside [1e-4, 10]"
```

**Assertions de convention.** Vérification que les conventions sont cohérentes entre modules.

```python
# Assertion : les rendements sont en log (pas arithmétiques)
# Vérification indirecte : log-returns sum ≈ period return
assert abs(returns.sum() - np.log(prices[-1]/prices[0])) < 1e-10, \
    "Returns are not log-returns — sum does not equal log(P_T/P_0)"
```

**Assertions de cohérence mathématique.** Vérification des identités mathématiques inter-modules.

```python
# Assertion : la covariance factorielle reconstruit la covariance asset
Sigma_reconstructed = B_A @ Sigma_z @ B_A.T + D_eps
Sigma_sample = np.cov(returns.T)
relative_error = np.linalg.norm(Sigma_reconstructed - Sigma_sample) / np.linalg.norm(Sigma_sample)
assert relative_error < 0.5, f"Factor model reconstruction error {relative_error:.2f} > 0.5"
```

**Assertions d'intégrité temporelle.** Spécifiques aux projets financiers : vérification de l'absence de look-ahead.

```python
# Assertion : aucune donnée future dans le calcul
assert all(train_dates < embargo_start), "Training data leaks past embargo"
assert all(test_dates > embargo_end), "Test data starts before embargo ends"
```

#### B.6.3 Organisation

Les assertions sont organisées en trois niveaux :

1. **Assertions unitaires** — internes à un module, incluses dans les tests du module.
2. **Assertions d'interface** — vérifiées aux points de jonction entre modules, incluses dans les tests d'intégration.
3. **Assertions globales** — propriétés end-to-end du pipeline, incluses dans les tests système.

Chaque assertion référence l'invariant (A.3) qu'elle vérifie, et les modules qu'elle concerne.

### B.7 Génération de l'ISD structuré

#### B.7.1 Objectif

Produire le document final : l'ISD, qui est la compilation de tout le travail précédent en un format directement consommable par les agents.

#### B.7.2 Structure de l'ISD

L'ISD est un ensemble de fichiers Markdown, un par module plus un fichier global :

```
docs/isd/
├── 00_global.md           # Conventions, invariants, glossaire
├── 01_data_pipeline.md    # Module MOD-001
├── 02_vae_architecture.md # Module MOD-002
├── ...
└── XX_integration.md      # Module d'intégration
```

**Le fichier global** (`00_global.md`) contient :
- Les conventions de nommage, d'indexation, et de notation.
- Le glossaire des termes et symboles.
- La liste complète des invariants (extraite de A.3).
- Le graphe de dépendances (extrait de B.3).
- Le plan de phases (extrait de B.4).
- Les critères d'acceptation globaux (extraits de A.4).

**Chaque fichier de module** contient les éléments détaillés dans C.3 (structure d'une section ISD consommable par un agent). Le point crucial : chaque fichier est **auto-suffisant** — un agent qui ne lit que ce fichier et le fichier global doit disposer de toute l'information nécessaire pour implémenter le module.

#### B.7.3 Règle d'auto-suffisance

L'auto-suffisance est la propriété la plus importante de l'ISD et la plus difficile à obtenir. Elle signifie :

- **Pas de renvoi au DVT.** Si le DVT dit "voir Section 4.7 pour la formule de rescaling", l'ISD doit reproduire la formule intégralement dans la section du module qui l'utilise.
- **Pas de renvoi inter-modules.** Si le module B dépend de la sortie du module A, l'ISD du module B doit spécifier la shape, le type, et les conventions de cette sortie — même si c'est redondant avec l'ISD du module A. La redondance est le prix de l'auto-suffisance.
- **Formules complètes.** Chaque formule mathématique est écrite avec ses variables définies localement, ses unités, ses conventions d'indexation, et ses cas limites.

Le coût de la redondance (volume de texte supérieur) est très inférieur au coût d'une erreur causée par un agent qui manque un renvoi.

### B.8 Validation humaine et itération

#### B.8.1 Points de validation obligatoires

| Étape | Artefact à valider | Question de validation |
|-------|-------------------|----------------------|
| B.1 | Topologie | "Tous les composants et couplages sont-ils représentés ?" |
| B.2 | Modules | "Chaque module est-il implémentable isolément ?" |
| B.3 | DAG | "Les dépendances et la parallelisabilité sont-elles correctes ?" |
| B.4 | Classification | "Le mode d'orchestration est-il approprié pour chaque module ?" |
| B.5 | Densité | "Les estimations de contexte sont-elles réalistes ?" |
| B.6 | Assertions | "Les contrats d'interface capturent-ils les invariants critiques ?" |
| B.7 | ISD | "Chaque section est-elle auto-suffisante ?" |

#### B.8.2 Critère de complétude de la Partie B

L'ISD est considéré comme complet lorsque le **test de l'agent naïf** est satisfait :

> Un agent Claude Code, disposant uniquement du fichier global et du fichier de son module, peut implémenter le module sans poser de question de clarification et en produisant du code qui passe toutes les assertions d'interface.

Ce test peut être vérifié empiriquement en demandant à un subagent d'implémenter un module non-critique (e.g., un benchmark simple) à partir de l'ISD seul, puis en vérifiant la qualité du résultat.

### B.9 Livrables de la Partie B — Checklist

- [ ] Topologie validée (B.1)
- [ ] Décomposition modulaire validée (B.2)
- [ ] DAG de dépendances validé (B.3)
- [ ] Classification séquentiel/parallèle validée (B.4)
- [ ] Estimation de densité réaliste (B.5)
- [ ] Assertions inter-modules complètes (B.6)
- [ ] ISD structuré et auto-suffisant (B.7)
- [ ] Test de l'agent naïf passé sur au moins un module (B.8)

---

## Partie C — Agents autonomes : exécution orchestrée

Cette partie — la plus volumineuse — spécifie comment les agents consomment l'ISD pour produire le code. Elle couvre l'architecture d'orchestration, les protocoles de communication, les templates opérationnels, et les mécanismes de détection et correction d'erreur.

### C.1 Architecture d'orchestration

#### C.1.1 Rôles

L'architecture repose sur quatre rôles distincts :

**Lead (humain + session Claude Code).** Responsable de : l'exécution du plan de phases (B.4), la supervision de la task list, l'injection de corrections quand un subagent ou teammate dévie, la validation des outputs inter-phases, et les décisions de conception non-anticipées par l'ISD.

**Orchestrateur (session lead Claude Code).** La session principale qui coordonne l'exécution. L'orchestrateur :
- Crée les agent teams pour les phases parallèles.
- Assigne les subagents pour les phases séquentielles.
- Vérifie les assertions d'interface après chaque module.
- Décide du passage à la phase suivante (toutes les tâches de la phase courante sont complètes et validées).
- Escalade vers l'humain en cas de blocage ou d'ambiguïté.

**Builder (subagent ou teammate).** Implémente un module de code à partir de sa section ISD. Le builder ne prend pas de décisions de conception — il traduit la spécification en code. Si l'ISD est ambigu, le builder signale l'ambiguïté plutôt que de la résoudre silencieusement.

**Validator (subagent).** Vérifie le code produit par un builder. Le validator ne modifie pas le code — il produit un rapport de conformité. Deux types de validation :
- **Validation formelle.** Le code passe les assertions (B.6), les tests unitaires, et satisfait les contraintes de style.
- **Validation sémantique.** Le code implémente correctement la spécification mathématique de l'ISD. Cette validation est plus profonde : elle vérifie que les formules sont correctement traduites, que les cas limites sont gérés, et que les conventions sont respectées.

#### C.1.2 Topologie de communication

```
Humain ←→ Orchestrateur (session lead)
               ↓ spawn              ↓ spawn
         [Agent Team]          [Subagents]
         ┌──────────┐         ┌──────────┐
         │ Teammate1 │←→│ Teammate2 │    │ Builder  │→│ Validator │
         │ (MOD-001) │  │ (MOD-002) │    │ (MOD-003)│  │ (MOD-003) │
         └──────────┘  └──────────┘    └──────────┘  └──────────┘
              ↕ messages directs            ↑ rapport
              via inbox                     │
                                     Orchestrateur
```

Les teammates communiquent entre eux via le système d'inbox. Les subagents rapportent uniquement à l'orchestrateur. L'orchestrateur est le seul point de contact avec l'humain pour les escalades.

### C.2 Configuration du CLAUDE.md projet

Le `CLAUDE.md` est le fichier de contexte que toutes les instances Claude Code (orchestrateur, teammates, subagents) lisent au démarrage. Il doit être concis (< 5K tokens) et contenir uniquement les informations nécessaires à **tous** les agents indépendamment de leur rôle.

#### C.2.1 Structure recommandée

```markdown
# [Nom du projet]

## Contexte
[2-3 phrases décrivant le projet et son objectif]

## Architecture
[Diagramme ASCII du pipeline]

## Conventions critiques
- [Convention 1 : e.g., "Tous les rendements sont en log, pas arithmétiques"]
- [Convention 2 : e.g., "Les indices de dimension commencent à 0"]
- [Convention 3 : ...]

## Invariants — NE PAS VIOLER
- [INV-001 : description courte]
- [INV-002 : ...]

## Structure du code
- `src/module_xxx/` : [description]
- `tests/` : [convention de nommage des tests]

## Workflow
- Lire `docs/isd/00_global.md` et la section ISD de votre module AVANT de coder
- Implémenter les assertions d'interface AVANT le code métier
- Committer après chaque sous-tâche complète
- Ne pas modifier les fichiers d'autres modules

## Dépendances
- Python 3.11+
- [Liste des bibliothèques avec versions]

## Tests
- `pytest tests/unit/` pour les tests unitaires
- `pytest tests/integration/` pour les tests d'intégration
- Un module n'est complet que si tous ses tests passent
```

#### C.2.2 Ce que le CLAUDE.md ne doit PAS contenir

- Les spécifications détaillées des modules (celles-ci sont dans l'ISD).
- Les formules mathématiques complètes (elles sont dans les sections ISD).
- L'historique des décisions de conception (il est dans le DVT).
- Les instructions d'orchestration (elles sont dans les prompts des agents).

Le CLAUDE.md est un **index** et un **garde-fou**, pas un document de spécification.

### C.3 Structure d'une section ISD consommable par un agent

Chaque section de l'ISD (un fichier par module) suit un template rigide. La rigidité est intentionnelle : elle garantit que l'agent trouve l'information attendue à l'emplacement attendu, sans ambiguïté structurelle.

#### C.3.1 Template

```markdown
# Module [MOD-XXX] — [Nom du module]

## Métadonnées
- **Phase d'exécution :** [1/2/3/4]
- **Mode d'orchestration :** [lead_session / subagent / teammate]
- **Dépendances :** [MOD-YYY (type D), MOD-ZZZ (type I)]
- **Densité de contexte estimée :** [low/medium/high/very_high]
- **Fichiers à produire :** [`src/module_xxx/main.py`, `src/module_xxx/utils.py`, ...]
- **Fichiers de test :** [`tests/unit/test_module_xxx.py`, ...]

## Objectif
[1-2 paragraphes décrivant ce que le module fait, pourquoi il existe,
et quel rôle il joue dans le pipeline global. Suffisant pour qu'un agent
comprenne la finalité sans lire d'autres documents.]

## Entrées
[Pour chaque entrée :]
| Nom | Type | Shape | Convention | Source |
|-----|------|-------|------------|--------|
| `windows` | `torch.Tensor` | `(N, T, F)` | float32, z-scored per window | MOD-001 output |

## Sorties
[Même format que les entrées]

## Spécifications techniques

### Sous-tâche 1 : [Nom]
[Description complète incluant :]
- Formule mathématique (avec définition de toutes les variables)
- Pseudo-code ou signature de fonction
- Cas limites et leur traitement
- Paramètres (valeur par défaut, plage valide)

### Sous-tâche 2 : [Nom]
[...]

## Invariants applicables
[Liste des invariants (de A.3) qui concernent ce module,
avec la description complète — pas un renvoi]

- **INV-001 :** [Description complète de l'invariant, y compris la
  conséquence de violation et la méthode de détection]

## Assertions d'interface
[Code Python des assertions que le module doit satisfaire]
```python
# assertion_mod_xxx.py
def verify_module_xxx_output(output):
    assert output.shape == (n, K), f"..."
    assert (output >= lower_bound).all(), f"..."
```

## Pièges connus
[Liste explicite des erreurs probables qu'un agent peut commettre]
- "NE PAS utiliser MSE brut sans le facteur D — cela cause un posterior collapse"
- "La rescaling vol doit être faite PER-DATE pour l'estimation historique,
   mais à la DATE COURANTE pour la construction du portefeuille"

## Tests requis
[Liste des tests que le builder doit implémenter et faire passer]
1. `test_output_shape` : vérifie les shapes de sortie
2. `test_edge_cases` : vérifie le traitement des cas limites
3. `test_mathematical_identity` : vérifie une identité connue
4. `test_no_lookahead` : vérifie l'absence de look-ahead (si applicable)

## Critères de complétion
- [ ] Tous les fichiers listés sont créés
- [ ] Toutes les assertions d'interface passent
- [ ] Tous les tests requis passent
- [ ] Le code est documenté (docstrings)
- [ ] Aucun TODO non-résolu dans le code
```

#### C.3.2 Règles de rédaction

**Formules auto-suffisantes.** Chaque formule redéfinit ses variables localement. Exemple :

```
Mauvais : "Appliquer la formule de la Section 4.4 du DVT"

Bon : "La loss complète est :
  L = (D / (2σ²)) · L_recon_weighted + (D/2) · ln(σ²) + L_KL + λ_co(t) · L_co
  où :
  - D = T × F (nombre d'éléments par fenêtre, T=504 par défaut, F=2)
  - σ² = exp(log_sigma_sq), scalaire appris, clampé à [1e-4, 10], initialisé à 1.0
  - L_recon_weighted = (1/|B|) Σ_w γ_eff(w) · MSE(w)
  - γ_eff(w) = 1 + f_c(w) · (γ - 1), avec f_c ∈ [0,1] la fraction de jours en crise
  - L_KL = (1/N) Σ_i (1/2) Σ_k (μ²_ik + σ²_ik - ln(σ²_ik) - 1)
  - λ_co(t) suit le curriculum : Phase 1 (λ_max), Phase 2 (décroissance linéaire → 0), Phase 3 (0)"
```

**Pièges formulés comme interdictions.** Les pièges connus doivent être formulés comme des interdictions explicites, pas comme des avertissements vagues.

```
Mauvais : "Attention à la normalisation"
Bon : "NE PAS normaliser par 1/(2σ²) sans le facteur D. La normalisation correcte est D/(2σ²)."
```

**Pseudo-code avec signatures typées.** Les signatures de fonctions doivent spécifier les types et les shapes.

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

### C.4 Protocole builder-validator

#### C.4.1 Principe

Le pattern builder-validator sépare la production du code de sa vérification. Un agent qui a implémenté un module a développé des biais d'ancrage sur son propre code — il est moins susceptible de détecter ses propres erreurs qu'un agent frais.

#### C.4.2 Séquence

```
1. Orchestrateur assigne le module au Builder (subagent)
   → Prompt : section ISD complète + fichier global
   
2. Builder implémente le code
   → Produit : fichiers source + tests unitaires
   → Commit intermédiaire
   
3. Orchestrateur assigne le code au Validator (subagent distinct)
   → Prompt : section ISD + code du Builder + assertions d'interface
   → Le Validator ne modifie pas le code
   
4. Validator produit un rapport :
   a. PASS — toutes les assertions satisfaites, code conforme à l'ISD
   b. FAIL — liste des non-conformités avec références à l'ISD
   
5a. Si PASS → Orchestrateur marque le module comme complet
5b. Si FAIL → Orchestrateur renvoie le rapport au Builder pour correction
   → Retour à l'étape 2 (max 3 itérations, puis escalade humaine)
```

#### C.4.3 Prompt du Builder

```markdown
## Rôle
Tu es un Builder. Ta tâche est d'implémenter le module [MOD-XXX] en suivant
strictement la spécification ISD ci-dessous. Tu ne prends aucune décision
de conception — tu traduis la spécification en code.

## Règles
1. Lire INTÉGRALEMENT la section ISD avant d'écrire une seule ligne de code.
2. Implémenter les assertions d'interface EN PREMIER — elles définissent le contrat.
3. Implémenter les sous-tâches dans l'ordre spécifié par l'ISD.
4. Pour chaque sous-tâche, écrire le test AVANT l'implémentation (TDD).
5. Committer après chaque sous-tâche complète (message : "MOD-XXX: sous-tâche N — [description]").
6. Si l'ISD est ambigu sur un point, NE PAS deviner — signaler l'ambiguïté
   dans un commentaire `# AMBIGUITY: [description]` et continuer avec
   l'interprétation la plus conservatrice.
7. Ne pas optimiser prématurément. Privilégier la clarté et la conformité à l'ISD.
8. Vérifier chaque formule implémentée contre la formule de l'ISD — caractère par caractère.

## Spécification ISD
[Section ISD complète du module]

## Contexte global
[Contenu de docs/isd/00_global.md]
```

#### C.4.4 Prompt du Validator

```markdown
## Rôle
Tu es un Validator. Ta tâche est de vérifier que le code du module [MOD-XXX]
est conforme à la spécification ISD. Tu ne modifies PAS le code —
tu produis un rapport de conformité.

## Procédure de vérification
1. Pour chaque sous-tâche de l'ISD, vérifier que :
   a. Le code implémente la formule exacte spécifiée (vérifier terme par terme).
   b. Les types et shapes correspondent à la spécification.
   c. Les cas limites sont gérés comme spécifié.
   d. Les paramètres ont les valeurs par défaut et plages spécifiées.
   
2. Pour chaque invariant applicable, vérifier que :
   a. L'invariant est respecté dans le code.
   b. Une assertion le vérifie explicitement.
   
3. Pour chaque piège listé, vérifier que :
   a. Le code ne tombe PAS dans le piège.
   
4. Exécuter les tests et vérifier qu'ils passent tous.
   
5. Vérifier les commentaires `# AMBIGUITY:` et évaluer si l'interprétation
   choisie par le Builder est acceptable.

## Format du rapport
```yaml
module: MOD-XXX
status: PASS | FAIL
issues:
  - severity: CRITICAL | MAJOR | MINOR
    location: "fichier:ligne"
    isd_reference: "Sous-tâche N, formule X"
    description: "..."
    recommendation: "..."
ambiguities_found:
  - location: "fichier:ligne"
    builder_interpretation: "..."
    assessment: "acceptable | needs_clarification"
```

## Spécification ISD
[Section ISD complète du module]

## Code à vérifier
[Code produit par le Builder]
```

### C.5 Gestion des phases d'exécution

#### C.5.1 Protocole par phase

**Initialisation d'une phase parallèle (Agent Team).**

```
1. Orchestrateur crée l'agent team :
   "Créer un agent team '[nom_phase]' avec N teammates."
   
2. Pour chaque teammate, l'orchestrateur fournit :
   - La section ISD du module assigné
   - Le fichier global (00_global.md)
   - Les sorties des modules de phases précédentes (si dépendance D)
   - Les interfaces gelées (si dépendance I)
   
3. Chaque teammate implémente son module de manière autonome.

4. Synchronisation : l'orchestrateur attend que tous les teammates
   aient terminé (TaskList : tous les items completed).
   
5. Validation inter-modules :
   - Exécuter les assertions d'interface entre les modules de la phase.
   - Si des assertions échouent, identifier le module fautif et relancer.
   
6. Shutdown de l'agent team.
7. Commit de phase : "Phase N complete — modules MOD-XXX, MOD-YYY, MOD-ZZZ"
```

**Exécution d'une phase séquentielle (Subagents).**

```
Pour chaque module de la phase (dans l'ordre de dépendance) :
  1. Spawn Builder subagent avec le prompt C.4.3
  2. Attendre la complétion
  3. Spawn Validator subagent avec le prompt C.4.4
  4. Si FAIL : renvoyer au Builder (max 3 itérations)
  5. Si PASS après itérations : marquer le module comme complet
  6. Exécuter les assertions d'interface avec les modules précédents
  7. Commit : "MOD-XXX complete — validated"
```

#### C.5.2 Gestion des transitions inter-phases

Avant de passer de la phase $n$ à la phase $n+1$ :

1. **Vérifier la complétion.** Tous les modules de la phase $n$ sont marqués comme complets et validés.
2. **Exécuter les tests d'intégration inter-modules.** Les assertions d'interface entre les modules de la phase $n$ et ceux des phases précédentes passent.
3. **Checkpoint Git.** Tag : `phase-N-complete`.
4. **Rapport de phase.** L'orchestrateur produit un résumé : modules complétés, assertions passées, problèmes rencontrés, décisions prises.
5. **Validation humaine (optionnelle mais recommandée).** L'humain revoit le rapport et approuve le passage à la phase suivante.

### C.6 Conventions de communication inter-agents

#### C.6.1 Communication Teammate ↔ Teammate (Agent Team)

Les teammates communiquent via le système d'inbox pour :
- **Signaler une interface prête.** "Le module MOD-001 est implémenté. L'interface de sortie est `DataPipelineOutput` dans `src/data_pipeline/types.py`."
- **Signaler un conflit.** "Le module MOD-002 attend un input de shape $(T, F)$ mais MOD-001 produit $(F, T)$. Qui s'adapte ?"
- **Partager un utility.** "J'ai implémenté une fonction `z_score_per_window()` dans `src/utils/normalization.py` — réutilisable."

Les messages doivent être factuels et actionables. Pas de discussion ouverte — les teammates ne sont pas un canal de brainstorming.

#### C.6.2 Communication Builder/Validator → Orchestrateur (Subagent)

Les subagents rapportent à l'orchestrateur exclusivement via :
- **Le code produit** (commité dans Git).
- **Le rapport de validation** (format YAML standardisé, C.4.4).
- **Les signaux d'escalade** (commentaires `# AMBIGUITY:` dans le code ou message explicite "ESCALADE: [description du blocage]").

#### C.6.3 Communication Orchestrateur → Humain

L'orchestrateur escalade vers l'humain dans trois cas :
1. **Ambiguïté non-résolvable.** L'ISD ne spécifie pas un comportement et le Builder ne peut pas choisir de manière conservatrice.
2. **Échec de validation persistant.** Le cycle Builder → Validator a atteint 3 itérations sans convergence.
3. **Conflit inter-modules.** Deux modules produisent des sorties incompatibles malgré les contrats d'interface.

### C.7 Critères de complétion et gestion d'erreur

#### C.7.1 Critères de complétion d'un module

Un module est complet si et seulement si :
1. Tous les fichiers listés dans la section ISD sont créés.
2. Tous les tests unitaires passent.
3. Toutes les assertions d'interface passent.
4. Le Validator a émis un rapport PASS (ou les issues MINOR/MAJOR ont été résolues).
5. Aucun commentaire `# AMBIGUITY:` non-résolu ne subsiste.
6. Le code est documenté (docstrings pour chaque fonction publique).

#### C.7.2 Critères de complétion d'une phase

1. Tous les modules de la phase sont complets.
2. Les tests d'intégration inter-modules passent.
3. Le checkpoint Git est créé.
4. Le rapport de phase est produit.

#### C.7.3 Critères de complétion du projet

1. Toutes les phases sont complètes.
2. Les tests end-to-end passent (le pipeline complet s'exécute).
3. Les critères d'acceptation globaux (A.4) sont satisfaits.
4. Le holdout final (si applicable) a été évalué.

#### C.7.4 Gestion d'erreur

**Erreur de compilation/runtime.** Le Builder corrige. Si le Builder ne peut pas corriger en 2 tentatives, l'orchestrateur escalade.

**Erreur d'assertion d'interface.** L'orchestrateur identifie le module responsable (en comparant la spécification ISD de l'interface avec les sorties réelles) et renvoie au Builder/Validator du module fautif.

**Erreur d'intégration inter-phases.** L'orchestrateur compare les interfaces réelles avec les contrats d'interface (B.6). Si le contrat est respecté mais l'intégration échoue, le contrat est probablement incomplet — escalade vers l'humain pour réviser le contrat.

**Dérive de spécification.** Un Builder a implémenté quelque chose qui passe les tests mais ne correspond pas à l'intention de l'ISD (détecté par le Validator sémantique). L'orchestrateur renvoie le rapport au Builder avec la référence ISD précise.

### C.8 Templates opérationnels

#### C.8.1 Template de définition de subagent

```yaml
# .claude/agents/builder.md
---
name: builder
description: "Implémente un module de code à partir d'une section ISD.
             Invoqué automatiquement pour les tâches d'implémentation."
tools: Read, Write, Edit, Bash, Glob, Grep
model: opus
---

Tu es un Builder spécialisé dans l'implémentation de modules à partir
de spécifications ISD. 

## Protocole
1. Lire intégralement la section ISD fournie.
2. Lire le fichier global (docs/isd/00_global.md).
3. Implémenter les assertions d'interface en premier.
4. Implémenter chaque sous-tâche dans l'ordre, avec test TDD.
5. Committer après chaque sous-tâche.
6. Signaler les ambiguïtés sans les résoudre.

## Interdictions
- Ne pas modifier les fichiers d'autres modules.
- Ne pas prendre de décisions de conception non-spécifiées par l'ISD.
- Ne pas optimiser prématurément.
```

```yaml
# .claude/agents/validator.md
---
name: validator
description: "Vérifie la conformité du code d'un module par rapport à sa
             section ISD. Invoqué après chaque implémentation de module."
tools: Read, Bash, Glob, Grep
model: opus
---

Tu es un Validator. Tu vérifies la conformité du code par rapport à l'ISD
sans modifier le code.

## Protocole
1. Lire la section ISD et le code produit.
2. Vérifier chaque sous-tâche (formule, types, shapes, cas limites).
3. Exécuter les tests.
4. Produire un rapport YAML standardisé.

## Interdictions
- Ne pas modifier le code source.
- Ne pas émettre d'opinion sur les choix de conception (ils viennent de l'ISD).
```

#### C.8.2 Template de prompt orchestrateur pour une phase parallèle

```markdown
## Phase [N] — [Nom de la phase]

### Modules à implémenter en parallèle :
- MOD-XXX ([teammate_name_1]) : [description courte]
- MOD-YYY ([teammate_name_2]) : [description courte]
- MOD-ZZZ ([teammate_name_3]) : [description courte]

### Instructions :
Créer un agent team "[phase_name]" avec 3 teammates.

Assigner à chaque teammate :
- Sa section ISD (docs/isd/XX_module_name.md)
- Le fichier global (docs/isd/00_global.md)
- Les sorties des phases précédentes : [liste des fichiers/interfaces disponibles]

Chaque teammate suit le protocole Builder (lire ISD → assertions → sous-tâches → tests → commit).

### Critères de synchronisation :
- Tous les modules implémentés et testés.
- Assertions d'interface inter-modules vérifiées.
- Tag Git : `phase-N-complete`.
```

#### C.8.3 Template de prompt orchestrateur pour une phase séquentielle

```markdown
## Phase [N] — [Nom de la phase]

### Ordre d'exécution :
1. MOD-XXX : [description courte]
2. MOD-YYY : [description courte] — dépend de MOD-XXX
3. MOD-ZZZ : [description courte] — dépend de MOD-YYY

### Pour chaque module :
1. Spawn Builder (subagent) avec :
   - Section ISD : docs/isd/XX_module_name.md
   - Fichier global : docs/isd/00_global.md
   - Sorties des modules précédents de cette phase
   
2. Attendre la complétion du Builder.

3. Spawn Validator (subagent) avec :
   - Section ISD (même)
   - Code produit par le Builder
   
4. Évaluer le rapport du Validator :
   - PASS → passer au module suivant
   - FAIL → renvoyer au Builder (max 3 itérations)
   - 3 échecs → ESCALADE vers l'humain

5. Vérifier les assertions d'interface avec les modules précédents.

6. Commit : "MOD-XXX validated"
```

### C.9 Anti-patterns et diagnostics

#### C.9.1 Anti-patterns courants

**Le Builder qui improvise.** Symptôme : du code qui "fonctionne" mais ne correspond pas à l'ISD (algorithme différent, paramètres non-spécifiés, heuristiques ajoutées). Cause : ISD insuffisamment détaillé ou Builder qui ne lit pas complètement la spécification. Remède : renforcer la section "Pièges connus" de l'ISD ; ajouter des assertions qui vérifient le comportement attendu, pas seulement les sorties.

**La validation superficielle.** Symptôme : le Validator émet PASS mais des bugs subsistent. Cause : le Validator vérifie uniquement que les tests passent sans vérifier la conformité sémantique (les tests eux-mêmes peuvent être insuffisants). Remède : exiger que le rapport du Validator liste explicitement chaque sous-tâche vérifiée avec la référence ISD.

**Le couplage fantôme.** Symptôme : un module passe ses tests unitaires mais l'intégration échoue. Cause : un couplage entre modules n'a pas été identifié en B.1 (degré de couplage sous-estimé). Remède : ajouter l'invariant manquant, créer l'assertion d'interface, relancer le module fautif.

**L'inflation de contexte.** Symptôme : un teammate ou subagent produit du code de qualité décroissante vers la fin de son module. Cause : le volume de spécification + code dépasse la capacité effective de la fenêtre de contexte. Remède : scinder le module si possible ; sinon, utiliser une session lead avec l'humain pour les parties critiques.

**La divergence de convention.** Symptôme : deux modules produisent des résultats numériquement différents pour la même opération (e.g., covariance calculée différemment). Cause : convention non-spécifiée dans le fichier global. Remède : ajouter la convention au CLAUDE.md et aux assertions d'interface.

#### C.9.2 Diagnostics de santé du projet

| Signal | Signification | Action |
|--------|--------------|--------|
| > 3 cycles Builder-Validator sur un module | ISD ambigu ou incomplet | Réviser la section ISD avec l'humain |
| Tests d'intégration échouent après phase complète | Contrats d'interface insuffisants | Ajouter des assertions inter-modules |
| Teammates communiquent excessivement | Modules trop couplés pour la parallélisation | Repasser en mode séquentiel |
| Commentaires `# AMBIGUITY:` fréquents | DVT ou ISD incomplet | Session de complétion DVT/ISD avec l'humain |
| Code produit significativement plus long que prévu | Le Builder ajoute de la logique non-spécifiée | Revue de conformité ISD |

### C.10 Livrables de la Partie C — Checklist

- [ ] Architecture d'orchestration définie (C.1)
- [ ] CLAUDE.md du projet rédigé (C.2)
- [ ] Sections ISD complètes et auto-suffisantes (C.3)
- [ ] Subagents Builder et Validator configurés (C.4, C.8.1)
- [ ] Prompts de phase préparés (C.5, C.8.2, C.8.3)
- [ ] Conventions de communication documentées (C.6)
- [ ] Critères de complétion formalisés à chaque niveau (C.7)
- [ ] Liste des anti-patterns partagée avec l'orchestrateur (C.9)
- [ ] Exécution complète des phases
- [ ] Tests end-to-end passés
- [ ] Critères d'acceptation globaux satisfaits

---

## Annexe — Glossaire

| Terme | Définition |
|-------|-----------|
| **DVT** | Document de Vision Technique — document source rédigé par l'utilisateur décrivant le projet |
| **ISD** | Implementation Specification Document — compilation du DVT en spécifications auto-suffisantes pour agents |
| **Module** | Unité de code implémentable et testable indépendamment |
| **Composant fonctionnel** | Unité logique du pipeline (peut ne pas correspondre 1:1 à un module) |
| **Invariant** | Propriété du système qui doit être vraie à tout moment |
| **Assertion d'interface** | Test formel vérifiable vérifiant un contrat entre modules |
| **Builder** | Agent qui implémente un module à partir de l'ISD |
| **Validator** | Agent qui vérifie la conformité du code par rapport à l'ISD |
| **Orchestrateur** | Session lead qui coordonne l'exécution des phases |
| **Lead session** | Session Claude Code directe, supervisée par l'humain |
| **Subagent** | Agent focalisé qui rapporte au lead, sans communication inter-agents |
| **Teammate** | Instance Claude Code autonome dans un Agent Team, avec communication inter-agents |
| **DAG** | Directed Acyclic Graph — graphe orienté sans cycle |
| **Couplage** | Degré d'interdépendance entre deux composants (échelle 0–4) |
| **Densité de contexte** | Volume de spécification qu'un agent doit maintenir en mémoire pour implémenter un module |
| **Auto-suffisance** | Propriété d'une section ISD contenant toute l'information nécessaire sans renvoi externe |
| **Phase** | Groupe de modules exécutables (séquentiellement ou en parallèle) avant synchronisation |
| **Checkpoint** | Tag Git marquant la complétion validée d'une phase |

---

*Document de méthodologie générale — applicable à tout projet satisfaisant les critères de périmètre (Introduction). L'instanciation sur un projet spécifique produit le Document 2 : l'ISD opérationnel.*
