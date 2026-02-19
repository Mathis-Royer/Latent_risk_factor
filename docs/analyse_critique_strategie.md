# Analyse Critique de la Stratégie de Facteurs de Risque Latents

> **Date:** 2026-02-18
> **Objectif:** Répondre aux interrogations fondamentales sur la conception de la stratégie, avec une revue de littérature académique complète et une comparaison systématique avec l'implémentation actuelle.

---

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Question 1 : Risque Symétrique vs. Risque Baissier](#2-question-1--risque-symétrique-vs-risque-baissier)
3. [Question 2 : Diversification et Cardinalité du Portefeuille](#3-question-2--diversification-et-cardinalité-du-portefeuille)
4. [Question 3 : Caractérisation des Facteurs de Risque](#4-question-3--caractérisation-des-facteurs-de-risque)
5. [Synthèse et Recommandations](#5-synthèse-et-recommandations)

---

## 1. Introduction et Contexte

### Ce que fait la stratégie actuelle

La stratégie repose sur un principe fondamental clairement énoncé dans le document de vision technique (DVT v4.1) : **minimiser le risque par une diversification optimale**, et non prédire les rendements. Cette distinction est cruciale car elle détermine toute l'architecture du système.

Le processus se décompose en plusieurs étapes :

1. **Découverte des facteurs** : Un VAE (Variational Autoencoder) analyse les co-mouvements historiques des prix pour découvrir des facteurs de risque latents, sans biais a priori.

2. **Construction du modèle de risque** : Les expositions de chaque action aux K facteurs forment une matrice B (n actions × K facteurs), qui permet d'estimer une matrice de covariance des actifs.

3. **Optimisation du portefeuille** : L'objectif est de maximiser l'entropie de Shannon sur les contributions au risque de chaque facteur principal, tout en minimisant la variance totale du portefeuille.

La fonction objectif actuelle est :

**max_w [ -λ · w'Σw + α · H(w) ]**

où :
- λ est l'aversion au risque (pénalise la variance totale)
- α est le poids de l'entropie (récompense la diversification factorielle)
- H(w) est l'entropie de Shannon sur les contributions au risque des facteurs principaux

Cette formulation implique un compromis : l'entropie maximale est atteinte quand chaque facteur contribue également au risque total, ce qui signifie que les expositions aux facteurs volatils sont réduites.

---

## 2. Question 1 : Risque Symétrique vs. Risque Baissier

### L'interrogation

> La stratégie cherche à sous-pondérer les facteurs volatils, donc le portefeuille va tendre vers une performance neutre car il exclut la prime de risque. Il serait peut-être plus intéressant de considérer des métriques considérant uniquement le downside ?

### Ce que dit la littérature académique

#### Le débat sur la prime de risque baissier

L'idée d'utiliser le risque baissier plutôt que le risque symétrique trouve son origine dans une observation intuitive : les investisseurs ne sont pas indifférents entre gagner et perdre. Une volatilité "positive" (gains supérieurs aux attentes) est souhaitable, tandis qu'une volatilité "négative" (pertes) est redoutée.

**Ang, Chen et Xing (2006)** ont publié une étude influente montrant que les actions avec un "beta baissier" élevé (forte corrélation avec le marché en période de baisse) généraient une prime de rendement d'environ 6% par an supérieure aux prédictions du CAPM traditionnel. Leur raisonnement : les investisseurs averses aux pertes exigent une compensation supplémentaire pour détenir des actifs qui chutent précisément quand le marché chute.

Cependant, **Atilgan, Demirtas et Gunaydin (2020)** ont remis en cause ces conclusions de manière significative. En reproduisant l'étude avec des méthodologies plus rigoureuses, ils ont découvert que la prime de beta baissier **disparaît** lorsque :
- On utilise des rendements pondérés par la capitalisation (au lieu de pondérations égales)
- On étend la période d'échantillonnage au-delà de 2006
- On mesure le beta baissier de manière ex-ante (prédictive) plutôt qu'ex-post (rétrospective)
- On contrôle pour d'autres déterminants des rendements

Leur conclusion est frappante : "L'association positive entre les betas baissiers et les rendements disparaît lorsque les betas baissiers sont mesurés ex-ante plutôt qu'ex-post." En d'autres termes, même si historiquement les actions à fort beta baissier ont généré des rendements supérieurs, cette relation n'est pas exploitable de manière prospective car elle repose sur une information qui n'était pas disponible au moment de la décision d'investissement.

#### Les défis techniques de l'optimisation semi-variance

Au-delà du débat sur la prime de risque, l'utilisation de métriques de risque baissier pose des défis techniques considérables.

**Le problème de l'estimation** : La semi-variance (variance calculée uniquement sur les rendements négatifs) nécessite environ deux fois plus d'observations que la variance standard pour atteindre une précision équivalente. Pourquoi ? Parce qu'on utilise seulement une partie des données (les jours de baisse), ce qui réduit l'échantillon effectif. Avec 50 ans de données journalières, cela reste gérable. Mais si l'on conditionne en plus sur l'état du marché (baisse du marché), l'échantillon effectif diminue encore drastiquement.

**Le problème de la convexité** : Contrairement à la matrice de covariance standard qui est toujours semi-définie positive (ce qui garantit que l'optimisation trouve un minimum global), la matrice de semi-covariance n'a pas cette propriété. Le problème d'optimisation devient non-convexe, ce qui signifie que l'algorithme peut converger vers différentes solutions selon le point de départ, et ces solutions peuvent être des minima locaux sous-optimaux.

**Le problème de l'intégration factorielle** : La stratégie actuelle estime une covariance des facteurs latents (Σ_z), puis projette vers une covariance des actifs. Comment traduire ce processus avec une semi-covariance ? Faut-il estimer une semi-covariance des facteurs ? Des actifs ? Les deux ? La littérature n'offre pas de réponse claire à cette question, car la plupart des travaux sur le risque baissier portent sur des modèles à un facteur (le marché) ou sur des actifs individuels, pas sur des modèles factoriels multidimensionnels.

### Comment la stratégie actuelle gère-t-elle ce problème ?

La stratégie n'est pas aveugle au risque asymétrique, même si elle utilise une variance symétrique. Deux mécanismes compensatoires sont en place :

**1. La pondération de crise dans l'entraînement du VAE (gamma = 3.0)**

Le VAE accorde un poids triple aux fenêtres de crise (identifiées par un VIX élevé) lors de son entraînement. Cela signifie que l'encodeur apprend à capturer les structures de co-mouvement qui apparaissent en période de stress de marché, précisément quand les corrélations explosent et les diversifications échouent. Le terme de "co-movement loss" dans la fonction de perte force explicitement le modèle à bien reconstruire les corrélations croisées, avec une emphase triple sur les périodes de crise.

**2. La mémoire anti-cyclique**

La fenêtre d'entraînement est expansive (non glissante) : chaque fold walk-forward utilise tout l'historique disponible. Une action exposée à un facteur de contagion en 2008 conserve cette exposition dans son profil composite, même si le facteur est resté dormant depuis. Si ce facteur se réactive, le portefeuille est déjà diversifié contre lui.

### Pourquoi l'approche symétrique est défendable

**Argument 1 : Pas de prévision de rendement = pas de capture de prime**

La stratégie opère par défaut avec μ = 0 (pas de prévision de rendement). Elle ne cherche pas à capturer une quelconque prime de risque, qu'elle soit baissière ou autre. Son objectif est de construire un portefeuille dont la structure de risque est optimale, pas de prédire quels facteurs vont performer.

Si l'on passait à une semi-variance sans ajouter de signal de rendement, on obtiendrait simplement un portefeuille différent mais toujours sans vue directionnelle. Le changement de métrique de risque ne crée pas magiquement une capacité à capturer des primes.

**Argument 2 : La précision d'estimation prime sur la sophistication théorique**

L'article fondateur de DeMiguel, Garlappi et Uppal (2009) a démontré un résultat contre-intuitif : avec les fenêtres d'estimation typiques, un simple portefeuille équipondéré (1/N) bat la plupart des méthodes d'optimisation sophistiquées en termes de ratio de Sharpe hors échantillon. Pourquoi ? Parce que les gains théoriques de l'optimisation sont plus que compensés par les erreurs d'estimation.

Ce résultat s'applique a fortiori aux méthodes de risque baissier qui nécessitent plus de données pour des estimations équivalentes. Utiliser une semi-variance avec 50 ans de données pourrait donner des résultats moins stables qu'une variance standard sur la même période.

**Argument 3 : La pondération de crise capture l'essentiel**

L'objectif du risque baissier est de mieux modéliser les périodes où "tout va mal ensemble". La pondération gamma = 3.0 sur les fenêtres de crise fait exactement cela : elle force le VAE à accorder trois fois plus d'attention aux structures de dépendance en période de stress. C'est une forme de risque baissier implicite, intégrée au niveau de l'apprentissage des facteurs plutôt qu'au niveau de l'optimisation du portefeuille.

### Quand considérer le risque baissier

Le passage à une optimisation basée sur le risque baissier pourrait se justifier dans deux scénarios :

**Scénario 1 : Mode directionnel avec signal de rendement (μ ≠ 0)**

Si la stratégie est utilisée avec des prévisions de rendement externes (momentum, valorisation), alors la distinction entre risque symétrique et baissier devient pertinente. Clarke, de Silva et Thorley (2013) ont montré que la parité de risque factorielle produit des ratios de Sharpe sous-optimaux lorsque les espérances de rendement des facteurs diffèrent matériellement. Dans ce cas, un "Sortino-budgeted factor risk parity" pourrait être envisagé.

**Scénario 2 : Contrainte réglementaire ou mandat**

Certains mandats de gestion spécifient explicitement des contraintes de drawdown maximal ou de VaR. Dans ce cas, intégrer le risque baissier dans l'optimisation (plutôt que de le traiter comme une contrainte post-hoc) pourrait améliorer l'efficience.

### Conclusion sur la Question 1

**La stratégie actuelle est défendable** pour son objectif de diversification pure (μ = 0). La pondération de crise dans le VAE fournit une forme de sensibilité au risque baissier au niveau de la découverte des facteurs. Le passage à une optimisation semi-variance introduirait des complexités techniques (estimation, convexité, intégration factorielle) dont les bénéfices ne sont pas clairement établis, surtout en l'absence de prévision de rendement.

---

## 3. Question 2 : Diversification et Cardinalité du Portefeuille

### L'interrogation

> La stratégie n'inclut pas dans l'optimisation paramétrique la diversité / non-concentration sur quelques actions mais seulement comme malus. Les résultats de diagnostic montrent que seules quelques dizaines d'actions sont sélectionnées. Comment intégrer un maximum de diversité ? La stratégie ne devrait-elle pas augmenter l'exposition à un facteur en augmentant le nombre d'actions corrélées à ce facteur plutôt qu'en augmentant le poids des actions existantes ?

### Ce que fait la stratégie actuelle

L'implémentation actuelle gère la diversification à plusieurs niveaux, mais avec des mécanismes qui peuvent paradoxalement conduire à une concentration :

**Niveau 1 : Entropie factorielle (objectif principal)**

L'entropie H(w) est calculée sur les contributions au risque des facteurs principaux, pas sur les poids des actions individuelles. Cela signifie que l'optimiseur cherche à diversifier le risque **entre facteurs**, pas nécessairement entre actions. Un portefeuille de 30 actions peut avoir une entropie factorielle élevée si ces 30 actions couvrent bien tous les facteurs.

**Niveau 2 : Contraintes de concentration (pénalité et plafond)**

- Plafond dur : w_max = 5% par action
- Seuil souple : w_bar = 3% avec pénalité quadratique φ·Σmax(0, w_i - w_bar)²
- Position minimale : w_min = 0.1% (semi-continue : soit zéro, soit au moins 0.1%)

**Niveau 3 : Enforcement de cardinalité (post-optimisation)**

Après l'optimisation SCA, les positions sous le seuil w_min sont éliminées de manière séquentielle, en préservant celles qui contribuent le plus à l'entropie. C'est une procédure de nettoyage, pas une optimisation de la cardinalité.

**Le problème observé** : Ces mécanismes ne garantissent pas un nombre minimum d'actions. L'entropie factorielle peut être satisfaite avec peu d'actions si celles-ci sont bien positionnées dans l'espace factoriel. Les contraintes de concentration limitent le poids maximal mais pas le nombre minimal.

### Ce que dit la littérature académique

#### Combien d'actions pour diversifier le risque idiosyncratique ?

La question "combien d'actions faut-il pour être diversifié ?" a une longue histoire en finance.

**La vision classique** (Evans et Archer, 1968) suggérait que 8 à 10 actions suffisaient pour éliminer la majeure partie du risque idiosyncratique. Cette conclusion reposait sur l'observation que la variance d'un portefeuille équipondéré diminue rapidement avec les premières actions ajoutées, puis se stabilise.

**La vision moderne** est plus nuancée. Une étude publiée dans le Journal of Risk and Financial Management (2021) a réexaminé cette question avec des données contemporaines et conclu que :
- Le risque non-systématique a augmenté relativement à la variabilité totale du marché au cours des 30 dernières années
- Les corrélations entre actions ont diminué, soulignant le besoin de portefeuilles plus grands
- 18 mesures de performance différentes donnent des comptes optimaux d'actions très différents
- "Une quantité significative de risque idiosyncratique subsiste, même pour des portefeuilles avec un grand nombre d'actions"

#### Le résultat DeMiguel et ses implications

L'article de DeMiguel, Garlappi et Uppal (2009) contient un résultat quantitatif frappant :

"La fenêtre d'estimation nécessaire pour que la stratégie moyenne-variance et ses extensions surpassent le benchmark 1/N est d'environ 3000 mois pour un portefeuille de 25 actifs et d'environ 6000 mois pour un portefeuille de 50 actifs."

Avec 50 ans de données (600 mois), nous sommes loin des 3000-6000 mois requis. **Cela signifie que pour des portefeuilles de 50+ actions, les méthodes simples (équipondération, inverse de volatilité) dominent statistiquement les méthodes optimisées** à cause de l'erreur d'estimation.

Ce résultat a une implication directe pour notre stratégie : plus on augmente le nombre d'actions, plus il devient difficile de battre une simple équipondération. L'ajout de complexité (entropie factorielle, VAE) doit apporter un avantage suffisant pour compenser l'erreur d'estimation accrue.

#### Maximum Diversification Portfolio (MDP) de Choueifaty

Une alternative à l'entropie factorielle est le ratio de diversification proposé par Choueifaty et Coignard (2008) :

**DR = (Σ w_i σ_i) / σ_portefeuille**

Le numérateur est la moyenne pondérée des volatilités individuelles. Le dénominateur est la volatilité du portefeuille. Si toutes les actions étaient parfaitement corrélées, DR = 1. Plus les corrélations sont faibles, plus DR peut être élevé.

Le portefeuille MDP maximise ce ratio. Il présente une propriété intéressante : sous hypothèse de corrélations constantes, MDP coïncide avec le portefeuille de parité de risque. Mais en général, les deux approches diffèrent.

**Comparaison avec l'entropie factorielle** :
- L'entropie factorielle diversifie le risque entre facteurs latents
- Le ratio de diversification diversifie le risque entre actifs observés
- L'entropie ignore les corrélations inter-actifs directes ; elle les capture via les facteurs
- Le ratio de diversification opère directement sur les corrélations observées

Une méta-approche utilisant l'entropie quadratique de Rao unifie ces deux perspectives, mais ajoute de la complexité.

#### Hierarchical Risk Parity (HRP) de Lopez de Prado

Marcos Lopez de Prado (2016) a proposé une approche radicalement différente : le Hierarchical Risk Parity.

**Le problème qu'il résout** : Les méthodes classiques (Markowitz, minimum variance, parité de risque) nécessitent l'inversion de la matrice de covariance, qui devient instable quand le nombre d'actifs est grand ou quand la matrice est mal conditionnée.

**La solution HRP** :
1. Regrouper les actifs hiérarchiquement selon leur distance de corrélation
2. Réorganiser la matrice de covariance pour mettre les actifs similaires côte à côte
3. Allouer le risque par bissection récursive de l'arbre hiérarchique

**Avantages pour la cardinalité** :
- HRP ne nécessite pas d'inversion matricielle, donc reste stable avec des centaines d'actifs
- Il produit naturellement des portefeuilles avec de nombreuses positions
- Il atténue à la fois le risque idiosyncratique (diversification intra-cluster) et systématique (diversification inter-clusters)

### Le dilemme "profondeur vs. largeur"

L'interrogation originale pose une question fondamentale : pour augmenter l'exposition à un facteur, vaut-il mieux :
- **Profondeur** : Augmenter le poids des actions existantes exposées à ce facteur
- **Largeur** : Ajouter plus d'actions exposées à ce facteur

La réponse dépend de l'objectif :

**Arguments pour la profondeur** :
- Moins de turnover (transactions)
- Plus simple à gérer opérationnellement
- Moins d'erreur d'estimation (moins de poids à estimer)

**Arguments pour la largeur** :
- Moins de risque idiosyncratique (un événement spécifique à une action affecte moins le portefeuille)
- Plus de robustesse aux erreurs de classification factorielle
- Meilleure conformité à la "Loi Fondamentale de la Gestion Active" de Grinold (1989), qui établit que le ratio d'information est proportionnel à la racine carrée du nombre de paris indépendants

**La nuance importante** : Ajouter des actions corrélées au même facteur n'augmente pas vraiment le nombre de paris indépendants. Si les actions A et B ont toutes deux une forte exposition au facteur 1, ajouter B quand on a déjà A diversifie le risque idiosyncratique mais pas le risque factoriel. La "largeur" au sens de Grinold concerne les paris indépendants, pas le simple compte d'actions.

### Comment la stratégie actuelle devrait évoluer

#### Ce qui est déjà bien conçu

L'architecture actuelle de deux niveaux (entropie factorielle + contraintes de concentration) est théoriquement solide. L'entropie sur les facteurs principaux garantit qu'aucun facteur ne domine le risque ; les contraintes empêchent la concentration excessive sur quelques actions.

#### Ce qui manque

**1. Contrainte de cardinalité minimale**

Il n'existe actuellement aucune contrainte forçant un nombre minimum d'actions. L'optimiseur peut converger vers 30 actions si cela maximise l'objectif.

**Recommandation** : Ajouter une contrainte n_actions >= 100, justifiée par :
- DeMiguel montre que l'erreur d'estimation domine pour n < 100 avec 600 mois d'historique
- Le risque idiosyncratique décroît significativement jusqu'à ~100 actions
- C'est une contrainte simple, interprétable, sans complexité algorithmique

**2. Contrainte de largeur par facteur**

La stratégie actuelle ne garantit pas que chaque facteur important soit couvert par plusieurs actions. Un facteur pourrait être "porté" par une seule action à fort poids.

**Recommandation** : Pour les K_top facteurs les plus importants (par valeur propre), exiger au moins n_min actions avec une exposition significative (|B_ik| > seuil).

Exemple : Pour les 20 premiers facteurs, exiger >= 5 actions avec |B_ik| > 0.3.

Cette contrainte prévient la vulnérabilité "une-action-par-facteur" mentionnée dans le Principe 8 du document de philosophie stratégique.

**3. Composante MDP dans l'objectif**

L'entropie factorielle diversifie entre facteurs mais pas directement entre actifs. Ajouter une composante de ratio de diversification pourrait encourager un plus grand nombre de positions.

**Formulation proposée** :
- Actuel : max H(w) - λ w'Σw - φ P_conc(w)
- Proposé : max H(w) + β·DR(w) - λ w'Σw

où β pondère l'importance du ratio de diversification. Cette modification inciterait l'optimiseur à préférer des solutions avec plus d'actions à entropie factorielle égale.

### La question spécifique : augmenter via le nombre plutôt que le poids

> La stratégie ne devrait pas augmenter l'exposition à un facteur en augmentant la part des actions corrélées à ce facteur déjà sélectionnées mais en augmentant le nombre d'actions corrélées à ce facteur ?

Cette intuition est partiellement correcte mais nécessite une nuance :

**Pour augmenter l'exposition à un facteur** (quand c'est l'objectif), les deux méthodes sont mathématiquement équivalentes du point de vue factoriel. Si l'action A a une exposition de 0.8 au facteur 1 et l'action B une exposition de 0.4, alors :
- Doubler le poids de A augmente l'exposition factorielle de 0.8
- Ajouter B avec le même poids que A augmente l'exposition de 0.4

**Mais pour réduire le risque idiosyncratique**, ajouter B est toujours préférable (à exposition factorielle égale). C'est l'argument de la largeur.

**Le problème** : La stratégie actuelle ne cherche pas à augmenter l'exposition à un facteur ; elle cherche à égaliser les contributions au risque. Donc la question devient : "Comment égaliser les contributions tout en maximisant le nombre d'actions ?"

**La réponse** : Les contraintes de cardinalité minimale et de largeur par facteur, combinées à une pénalité de concentration efficace (φ > 0), forcent l'optimiseur à trouver des solutions avec plus d'actions. Le ratio de diversification en complément récompense explicitement la dispersion des poids.

### Conclusion sur la Question 2

La stratégie actuelle privilégie la diversification **factorielle** au détriment de la diversification **actionnariale**. Pour les 20,000 actions et 50 ans de données disponibles :

1. **Contrainte de cardinalité minimale (n >= 100)** : Justifiée par la littérature sur l'erreur d'estimation et le risque idiosyncratique.

2. **Contrainte de largeur par facteur** : Empêche la vulnérabilité d'un facteur porté par une seule action.

3. **Composante MDP optionnelle** : Pour encourager explicitement un plus grand nombre de positions.

4. **Prudence sur l'objectif** : Avec 20,000 actions, l'erreur d'estimation devient considérable. DeMiguel suggère que des méthodes simples (1/N, inverse vol) peuvent dominer. Le benchmark 1/N doit rester une référence sérieuse.

---

## 4. Question 3 : Caractérisation des Facteurs de Risque

### L'interrogation

> Comment traiter les différents facteurs identifiés ? Car ils ont des périmètres différents ? Parler de "force" d'un facteur n'a pas de sens car elle dépend des événements passés réalisés et de leur force. Quelles autres dimensions pourraient décrire un facteur de risque ? La fréquence ? Le but n'est pas de corriger la stratégie par des couches supplémentaires mais de repenser proprement les fondements si nécessaire.

### Ce que fait la stratégie actuelle

La caractérisation des facteurs est actuellement minimale :

**Dimension 1 : Actif vs. Inactif (AU)**
- Un facteur est "actif" si sa divergence KL > 0.01 nats
- Cela distingue les dimensions latentes utilisées par le VAE de celles qui sont restées à leur prior N(0,1)
- C'est un critère binaire, sans gradation

**Dimension 2 : Valeur propre (force)**
- Après diagonalisation de Σ_z, les facteurs principaux sont ordonnés par valeur propre décroissante
- La valeur propre λ_k représente la variance du facteur k
- Les facteurs à forte valeur propre contribuent plus au risque total

**Ce qui manque** : Une caractérisation multidimensionnelle permettant de comprendre la nature de chaque facteur et d'adapter le traitement en conséquence.

### Ce que dit la littérature académique

#### Les dimensions de qualité d'un facteur

**Lettau et Pelger (2020)** ont développé une méthode d'estimation des facteurs latents de pricing (RP-PCA) qui surpasse significativement le PCA standard. Leur analyse révèle plusieurs dimensions importantes :

**1. Persistance (demi-vie)**

La persistance mesure combien de temps les rendements d'un facteur restent autocorrélés. Un facteur très persistant (demi-vie longue) maintient sa direction pendant des semaines ou des mois ; un facteur peu persistant (demi-vie courte) change de signe fréquemment.

**Formule** : demi-vie = ln(2) / ln(|ρ₁|), où ρ₁ est l'autocorrélation d'ordre 1 des rendements du facteur.

**Pourquoi c'est important** : Un facteur avec une demi-vie de 3 jours ne peut pas être exploité avec un rebalancement mensuel. Si le portefeuille se rebalance tous les 21 jours mais que les expositions factorielles pertinentes changent tous les 3 jours, le portefeuille sera systématiquement en retard.

**Application à la stratégie** : Les facteurs à faible persistance pourraient être exclus de l'entropie ou pondérés différemment, car ils sont difficiles à exploiter au niveau de risque.

**2. Périmètre (breadth / nombre d'actions affectées)**

Le périmètre mesure combien d'actions sont significativement exposées à un facteur.

**Giglio et al. (2021)** distinguent :
- **Facteurs forts** : Affectent essentiellement toutes les actions (ex: le facteur marché)
- **Facteurs faibles** : Affectent seulement un sous-ensemble (ex: facteur sectoriel, risque de queue)

Leur conclusion : "Beaucoup de facteurs de pricing sont faibles" — plus difficiles à détecter statistiquement mais potentiellement plus profitables car moins arbitrés.

**Application à la stratégie** : Le VAE peut découvrir à la fois des facteurs forts et faibles. Les facteurs faibles nécessitent une détection explicite car ils peuvent être noyés dans le bruit. La contrainte de "largeur par facteur" (≥5 actions par facteur important) devient problématique pour les facteurs faibles par définition.

**3. Ratio signal/bruit (gap de valeurs propres)**

**Onatski (2010)** a développé des tests statistiques basés sur les écarts entre valeurs propres adjacentes pour déterminer le nombre de "vrais" facteurs.

L'intuition : Si les K premiers facteurs capturent un vrai signal et les suivants ne sont que du bruit, il devrait y avoir un "gap" visible entre λ_K et λ_{K+1}. Les valeurs propres du bruit suivent une distribution de Tracy-Widom et sont donc prévisibles ; les valeurs propres du signal sont significativement plus grandes.

**Application à la stratégie** : Le critère AU (KL > 0.01 nats) est une heuristique non-statistique. Les tests de Onatski ou Bai-Ng IC2 fourniraient une validation formelle du nombre de facteurs découverts.

**4. Stabilité (constance des expositions dans le temps)**

Un facteur stable maintient des expositions similaires d'un fold à l'autre ; un facteur instable voit ses expositions changer drastiquement.

**Mesure** : corr(B^(t), B^(t-1)), la corrélation entre les expositions d'un fold et celles du fold précédent.

**Pourquoi c'est important** : Un facteur très instable peut être du bruit échantillonnal plutôt qu'une vraie structure de risque. Il peut aussi refléter un changement de régime (fusion de deux facteurs, disparition d'un secteur).

Le document de philosophie stratégique mentionne trois catégories de facteurs par persistance temporelle :
- **Facteurs structurels** (industrie, géographie) : stabilité > 0.90
- **Facteurs de style** (levier, taille, momentum) : demi-vies > 25 mois
- **Facteurs épisodiques** (contagion de crise) : dormants pendant des années, puis soudainement dominants

**5. Crowding (concentration des investisseurs)**

Un facteur "crowded" est un facteur sur lequel de nombreux investisseurs institutionnels ont pris position similairement. Cela compresse la prime de risque et augmente le risque de crash (tout le monde veut sortir en même temps).

**Mesure** : Corrélation des rendements du facteur avec des stratégies connues pour être crowded (momentum, value, low-vol).

**Limitation** : Non mesurable directement depuis les prix seuls ; nécessite des données de positioning ou de flux.

**6. Dépendance de queue (corrélations non-linéaires)**

Deux facteurs peuvent paraître non-corrélés en temps normal mais devenir très corrélés dans les queues de distribution (crises).

**Mesure** : Coefficient de dépendance de queue, analyse par copules.

**Application à la stratégie** : La pondération de crise (gamma = 3.0) capture partiellement cet effet en accordant plus de poids aux périodes de stress, mais ne le mesure pas explicitement.

### Ce qui détermine les K facteurs du VAE

L'interrogation demande si les K facteurs sont "forcément les plus récurrents ? Les plus forts ? Avec les plus gros périmètres ?"

**Réponse** : Les trois dimensions jouent un rôle, mais avec des pondérations différentes selon l'architecture.

Le VAE minimise une fonction de perte composite :
- **Reconstruction** : Force le modèle à capturer les co-mouvements qui expliquent la variance des rendements
- **KL régularisation** : Pousse les dimensions non-informatives vers le prior N(0,1), les "désactivant"

**Ce qui fait qu'un facteur survit à la régularisation KL** :
- **Récurrence** : Un facteur doit expliquer de la variance sur de nombreuses fenêtres (sinon le KL l'éteint)
- **Force** : Les facteurs à forte variance contribuent plus à la reconstruction et résistent donc mieux au KL
- **Périmètre** : Un facteur affectant beaucoup d'actions contribue plus à la reconstruction globale

**Empiriquement**, les facteurs découverts tendent à être :
1. Le facteur marché (plus forte valeur propre, périmètre universel, récurrence permanente)
2. Les facteurs sectoriels (variance modérée, périmètre moyen, récurrence permanente)
3. Les facteurs de style (variance modérée, périmètre large, évolution lente)
4. Les facteurs de crise (variance faible normalement, périmètre large en crise, récurrence épisodique)

**Point critique** : Les facteurs épisodiques (crise) peuvent être sous-représentés car ils n'expliquent de la variance que pendant quelques périodes. C'est pourquoi la pondération de crise (gamma = 3.0) est essentielle : elle amplifie l'importance des fenêtres de crise dans la fonction de perte.

### Proposition : un tableau de bord de qualité des facteurs

Pour caractériser proprement les facteurs sans ajouter de "couches correctives", je propose d'intégrer un tableau de bord diagnostique calculé pour chaque facteur après entraînement :

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Valeur propre** | λ_k de Σ_z | Contribution à la variance (force) |
| **Périmètre** | count(\|B_ik\| > 0.3) | Nombre d'actions affectées |
| **Persistance** | ln(2) / ln(\|autocorr(z_k)\|) | Demi-vie en jours |
| **Stabilité** | corr(B^(t)_k, B^(t-1)_k) | Constance entre folds |
| **Activité KL** | KL_k du VAE | Distance au prior (actif si > 0.01) |
| **Gap de valeur propre** | λ_k - λ_{k+1} | Séparation signal/bruit |

**Utilisation du tableau de bord** :
- Identifier les facteurs instables (stabilité < 0.5) pour les signaler comme suspects
- Identifier les facteurs à faible persistance (< fréquence de rebalancement) pour évaluer leur exploitabilité
- Valider le compte AU par les tests Onatski/Bai-Ng
- Distinguer programmatiquement les facteurs structurels, de style, et épisodiques

**Ce que cela ne nécessite pas** : Modifier l'optimisation du portefeuille. Le tableau de bord est diagnostique, pas prescriptif. La fonction objectif (entropie factorielle) reste inchangée.

### Faut-il pondérer l'entropie par la qualité des facteurs ?

Une question naturelle est : faut-il modifier l'entropie pour donner plus de poids aux facteurs "de qualité" (persistants, stables, à large périmètre) ?

**Arguments pour** :
- Les facteurs à faible persistance ne peuvent pas être exploités au niveau du risque avec un rebalancement mensuel
- Les facteurs instables peuvent être du bruit
- Accorder le même poids à tous les facteurs traite un facteur éphémère comme le facteur marché

**Arguments contre** :
- Le risque existe indépendamment de notre capacité à l'exploiter ; un facteur à faible persistance peut quand même faire perdre de l'argent
- La pondération par la qualité introduit des hyperparamètres arbitraires
- Le principe de diversification stipule justement de ne pas faire de paris sur quels facteurs "comptent"

**Ma recommandation** : Garder l'entropie non-pondérée (tous les facteurs AU comptent également) mais utiliser le tableau de bord pour :
- Signaler des anomalies (trop peu de facteurs stables, persistance moyenne trop faible)
- Ajuster la fréquence de rebalancement si la persistance moyenne des facteurs importants est courte
- Documenter la composition du risque (X% structurel, Y% style, Z% épisodique)

---

## 5. Synthèse et Recommandations

### Les trois questions et leurs réponses

**Question 1 : Risque symétrique vs. baissier**

La stratégie actuelle utilisant une covariance symétrique est défendable pour un objectif de diversification pure (μ = 0). La pondération de crise (gamma = 3.0) dans le VAE capture implicitement les structures de dépendance baissières. Le passage à une optimisation semi-variance introduirait des complexités (estimation, convexité, intégration factorielle) sans bénéfice clair prouvé, surtout compte tenu du débat académique sur la prime de beta baissier (Ang 2006 vs. Atilgan 2020).

**Recommandation** : Conserver l'approche actuelle. Considérer le risque baissier explicite uniquement si la stratégie passe en mode directionnel (μ ≠ 0) avec des signaux de rendement.

**Question 2 : Diversification et cardinalité**

La stratégie diversifie au niveau factoriel mais ne garantit pas un nombre minimum d'actions. Avec 20,000 actions disponibles, l'erreur d'estimation devient critique (DeMiguel 2009). Les mécanismes actuels (entropie + contraintes de concentration) peuvent converger vers seulement 30-50 positions.

**Recommandations** :
1. Ajouter une contrainte de cardinalité minimale : n_actions >= 100
2. Ajouter une contrainte de largeur par facteur : ≥5 actions avec |B_ik| > 0.3 pour les 20 premiers facteurs
3. Optionnel : Intégrer une composante MDP (ratio de diversification) dans l'objectif
4. Maintenir le benchmark 1/N comme référence sérieuse

**Question 3 : Caractérisation des facteurs**

Les facteurs découverts par le VAE sont actuellement caractérisés seulement par leur statut actif/inactif (KL > 0.01) et leur valeur propre. La littérature (Lettau & Pelger 2020, Onatski 2010, Giglio et al. 2021) identifie des dimensions additionnelles : persistance, périmètre, stabilité, gap de valeurs propres.

**Recommandation** : Implémenter un tableau de bord de qualité des facteurs (diagnostique, pas prescriptif). Utiliser les métriques pour valider le compte AU, signaler des anomalies, et documenter la composition du risque. Ne pas modifier la fonction objectif.

### Principes directeurs pour toute modification

Le document de philosophie stratégique (strategy_philosophy.md) établit 10 principes fondateurs. Toute modification doit les respecter :

1. **Diversifier le risque, ne pas prédire les rendements** : Les ajouts proposés (cardinalité, tableau de bord) ne violent pas ce principe.

2. **Laisser les données révéler les facteurs** : Le tableau de bord caractérise les facteurs découverts, il ne les impose pas a priori.

3. **Mémoire des risques dormants** : La contrainte de cardinalité et de largeur par facteur renforce ce principe en s'assurant que les facteurs épisodiques soient couverts par plusieurs actions.

4. **Égaliser les contributions au risque** : L'entropie factorielle reste la fonction objectif principale.

5. **Validation walk-forward rigoureuse** : Toute modification doit prouver son bénéfice OOS sur les 34 folds.

### Ce qui NE devrait PAS être fait

- **Passer à une semi-variance** sans justification empirique claire et sans résoudre les problèmes techniques d'intégration factorielle
- **Pondérer l'entropie par la "qualité" des facteurs** sans critères objectifs et validés
- **Abandonner l'entropie factorielle pour le ratio de diversification** qui opère au niveau des actifs et ignore la structure latente
- **Ajouter des couches de ML pour "corriger" les défauts** perçus — la complexité doit être justifiée par un gain OOS mesurable

### Plan de validation

Pour toute modification implémentée, le protocole de validation doit être :

1. **Walk-forward complet** sur les 34 folds avec et sans la modification
2. **Tests statistiques** : Wilcoxon signed-rank pour comparer les métriques appariées par fold
3. **Benchmark 1/N** : La modification doit maintenir ou améliorer l'avantage sur 1/N
4. **Benchmark PCA** : La modification ne doit pas réduire l'avantage du VAE sur le PCA factor risk parity

---

## Références Académiques

### Risque baissier
- [Ang, Chen & Xing (2006)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=641843) - Downside Risk, Review of Financial Studies
- [Atilgan, Demirtas & Gunaydin (2020)](https://onlinelibrary.wiley.com/doi/abs/10.1111/eufm.12258) - European Financial Management
- [Palomar - Downside Risk Portfolios](https://bookdown.org/palomar/portfoliooptimizationbook/10.3-downside-risk-portfolios.html)

### Diversification et cardinalité
- [DeMiguel, Garlappi & Uppal (2009)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1376199) - Optimal vs. Naive Diversification, RFS
- [Choueifaty & Coignard (2008)](https://www.tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf) - Maximum Diversification, JPM
- [Lopez de Prado (2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678) - Hierarchical Risk Parity, SSRN
- [JRFM 2021](https://www.mdpi.com/1911-8074/14/11/551) - How Many Stocks for Diversification

### Caractérisation des facteurs
- [Lettau & Pelger (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0304407620300051) - Estimating Latent Asset-Pricing Factors, Journal of Econometrics
- [Onatski (2010)](https://academic.oup.com/jfec/article/23/1/nbad024/7271793) - Eigenvalue Tests for Latent Factors
- [Giglio et al. (2021)](https://finance.business.uconn.edu/wp-content/uploads/sites/723/2021/04/wf_22.pdf) - Test Assets and Weak Factors
- [Bai & Ng (2002)](https://sebkrantz.github.io/dfms/reference/ICr.html) - Factor Number Selection

### Crises et corrélations
- [Page & Panariello (2018)](https://www.tandfonline.com/doi/full/10.2469/faj.v74.n3.3) - When Diversification Fails, Financial Analysts Journal
- [Kritzman & Li (2010)](https://jpm.pm-research.com/content/36/2/33) - Skulls, Financial Turbulence, and Risk Management
- [Cont & Bouchaud (2000)](https://www.tandfonline.com/doi/abs/10.1080/713665670) - Herd Behavior and Aggregate Fluctuations

---

## Annexe : Questions de Suivi Approfondies

Cette annexe répond en détail aux questions de clarification soulevées après la première lecture.

---

### A.1 Existe-t-il une solution plus intuitive pour la cardinalité, sans contrainte fixe ?

#### L'interrogation

> N'y a-t-il pas de solution plus intuitive sans contrainte fixe mais plutôt optimale calculée par le modèle ? La composante MDP répond-elle entièrement à cette question ?

#### Réponse détaillée

**Le problème fondamental** : La littérature académique traite presque universellement la cardinalité comme une **contrainte exogène** (fixée par l'investisseur) plutôt qu'une **variable endogène** (optimisée par le modèle). Pourquoi ? Parce que le problème de cardinalité optimale est mathématiquement intractable.

Pour comprendre, considérons ce que signifierait "optimiser le nombre d'actions" :

1. On devrait définir un critère qui pénalise à la fois trop peu d'actions (risque idiosyncratique) et trop d'actions (erreur d'estimation, coûts de transaction)
2. Ce critère devrait être exprimé de manière différentiable pour être intégré à l'optimisation
3. Le problème reste NP-hard car la sélection d'un sous-ensemble optimal parmi n actions nécessite d'explorer 2^n combinaisons

**Ce que fait la composante MDP (ratio de diversification)** :

Le ratio de diversification DR = (Σ w_i σ_i) / σ_p récompense la dispersion des poids de manière indirecte. Si le portefeuille concentre ses poids sur peu d'actions à volatilité similaire, le numérateur et le dénominateur croissent proportionnellement, donc DR n'augmente pas. Mais si le portefeuille disperse ses poids sur des actions à corrélations imparfaites, le dénominateur (volatilité du portefeuille) diminue plus que le numérateur, donc DR augmente.

**Cependant, MDP ne résout pas entièrement le problème** :

1. MDP favorise la dispersion des poids, pas le nombre d'actions. Un portefeuille de 50 actions avec des poids de 2% chacune peut avoir un DR identique à un portefeuille de 100 actions avec des poids de 1% chacune — cela dépend uniquement des corrélations, pas du compte.

2. MDP peut même encourager la concentration si quelques actions ont des corrélations très faibles. Si l'action A a une corrélation de -0.3 avec le reste du portefeuille, MDP pourrait recommander un poids élevé sur A.

3. MDP n'incorpore pas le risque idiosyncratique de manière explicite. Il mesure le "bénéfice de diversification" des corrélations, pas la réduction du risque non-systématique.

**Existe-t-il une meilleure approche ?**

La littérature propose plusieurs pistes, mais aucune n'est clairement supérieure :

**Piste 1 : Pénalité sur le HHI (Herfindahl-Hirschman Index)**

Le HHI = Σ w_i² mesure la concentration des poids. Minimiser le HHI pousse vers l'équipondération. On pourrait ajouter un terme -ψ·HHI(w) à l'objectif.

Problème : HHI ne pénalise pas le faible nombre d'actions si celles-ci sont équipondérées. 30 actions à 3.33% chacune ont un HHI bas.

**Piste 2 : Pénalité sur le "nombre effectif de positions" (ENP)**

ENP = 1 / HHI = 1 / Σ w_i². C'est l'inverse du HHI, souvent appelé "nombre équivalent d'actifs".

On pourrait ajouter un terme +ψ·ENP(w), mais cela pose le même problème : ENP mesure la dispersion des poids, pas le compte d'actions non-nulles.

**Piste 3 : Contrainte semi-continue + coût marginal**

L'approche actuelle utilise w_i = 0 ou w_i >= w_min. On pourrait ajouter un coût marginal décroissant par action : chaque action ajoutée coûte moins que la précédente. Cela crée une incitation à ajouter des actions jusqu'à ce que le bénéfice marginal (réduction de risque idiosyncratique) égale le coût marginal.

Problème : Comment calibrer ce coût marginal ? C'est un hyperparamètre aussi arbitraire qu'un n_min fixe.

**Conclusion pragmatique** :

Il n'existe pas de solution "plus intuitive" qui détermine automatiquement le nombre optimal d'actions sans introduire d'hyperparamètres. Les contraintes fixes (n >= 100) ont l'avantage d'être simples, interprétables, et justifiées par la littérature (DeMiguel 2009 sur l'erreur d'estimation).

La composante MDP peut être un complément utile mais ne remplace pas une contrainte de cardinalité minimale. Elle encourage la dispersion des poids, pas le nombre d'actions.

---

### A.2 Pourquoi certaines actions ont des poids plus élevés que d'autres ?

#### L'interrogation

> Pourquoi certaines actions ont des poids plus faibles et d'autres plus élevés ?

#### Réponse détaillée

La question semble simple mais touche au cœur de ce que fait l'optimisation. Voici les mécanismes qui déterminent les poids :

**Mécanisme 1 : L'entropie factorielle favorise les actions "pivots"**

L'entropie de Shannon sur les contributions au risque factoriel est maximisée quand chaque facteur contribue également au risque total. Pour atteindre cet objectif, l'optimiseur doit ajuster les expositions du portefeuille à chaque facteur.

Une action "pivot" est une action qui permet d'augmenter l'exposition à un facteur sous-représenté sans augmenter l'exposition aux facteurs déjà sur-représentés. Ces actions reçoivent des poids plus élevés.

Exemple concret : Supposons que le facteur 1 (marché) contribue déjà trop au risque, et le facteur 5 (technologie de niche) pas assez. Une action qui a une forte exposition au facteur 5 et une faible exposition au facteur 1 devient précieuse pour rééquilibrer les contributions — elle recevra un poids élevé.

**Mécanisme 2 : Le terme de variance pénalise les actions volatiles**

Le terme -λ·w'Σw pénalise les actions à forte variance et à fortes corrélations avec les autres. À contribution factorielle égale, une action moins volatile recevra un poids plus élevé.

**Mécanisme 3 : Les contraintes créent des discontinuités**

- Le plafond w_max = 5% empêche toute action de dépasser 5%, même si l'entropie voudrait plus
- La pénalité de concentration φ·Σmax(0, w_i - w_bar)² pénalise les poids au-dessus de 3%
- La contrainte semi-continue (w_i = 0 ou w_i >= 0.1%) force les petites positions à disparaître

Ces contraintes créent une distribution bimodale des poids : des actions autour de 1-3% (celles qui contribuent à l'entropie), et des actions à exactement 0% (celles qui ont été éliminées par l'enforcement de cardinalité).

**Mécanisme 4 : L'enforcement de cardinalité élimine les actions marginales**

Après l'optimisation SCA, les positions sous w_min sont éliminées séquentiellement. L'algorithme élimine en priorité les actions dont la suppression coûte le moins d'entropie.

Cela crée une "sélection naturelle" : seules les actions qui contribuent significativement à l'entropie factorielle survivent. Les actions redondantes (qui n'apportent pas d'exposition factorielle unique) sont éliminées.

**En résumé, une action a un poids élevé si** :
1. Elle apporte une exposition unique à un facteur sous-représenté
2. Elle a une variance relativement faible
3. Elle n'est pas redondante avec d'autres actions du portefeuille

**Une action a un poids nul si** :
1. Son exposition factorielle est redondante avec d'autres actions
2. Sa variance est trop élevée par rapport à sa contribution
3. Elle a été éliminée lors de l'enforcement de cardinalité car marginale

---

### A.3 Que signifie "égaliser les contributions au risque" ?

#### L'interrogation

> Qu'est-ce que ça veut dire égaliser les contributions au risque ? Qu'est-ce que ça veut dire entre un facteur de risque de marché et un facteur niche ?

#### Réponse détaillée

C'est le concept central de la stratégie, et il est crucial de bien le comprendre.

**La contribution au risque d'un facteur**

Après diagonalisation de la covariance des facteurs (Σ_z = VΛV'), on obtient des facteurs principaux indépendants. La contribution au risque du facteur k est :

**c'_k = (β'_k)² × λ_k**

où :
- β'_k est l'exposition du portefeuille au facteur principal k (c'est le produit scalaire entre les poids w et les expositions des actions au facteur k)
- λ_k est la variance (valeur propre) du facteur k

**L'entropie normalise ces contributions** :

ĉ'_k = c'_k / Σ_j c'_j  (contribution normalisée, somme = 1)

H(w) = -Σ_k ĉ'_k × ln(ĉ'_k)  (entropie de Shannon)

L'entropie est maximale quand toutes les contributions normalisées sont égales : ĉ'_1 = ĉ'_2 = ... = ĉ'_AU = 1/AU.

**Ce que cela implique pour les expositions**

Si on veut c'_1 = c'_2 pour deux facteurs de variances différentes :

(β'_1)² × λ_1 = (β'_2)² × λ_2

Donc : |β'_1| / |β'_2| = √(λ_2 / λ_1)

**Exemple concret : marché vs. niche**

Supposons :
- Facteur marché : λ_marché = 0.05 (forte variance, ce facteur "bouge beaucoup")
- Facteur niche : λ_niche = 0.001 (faible variance, ce facteur est plus stable)

Pour égaliser les contributions au risque :

|β'_niche| / |β'_marché| = √(0.05 / 0.001) = √50 ≈ 7.07

Le portefeuille doit avoir une exposition **7 fois plus grande** au facteur niche qu'au facteur marché !

**L'intuition derrière cette logique**

Pourquoi vouloir une exposition plus grande aux facteurs faibles ? Parce que l'objectif est d'égaliser les **contributions au risque**, pas les **expositions**.

Pensez-y ainsi : si vous pariez 10€ sur une roulette (probabilité 1/37 de gagner) et 10€ sur un pile-ou-face (probabilité 1/2), votre "exposition" est égale mais votre "risque" est très différent. Pour égaliser le risque, vous devriez parier moins sur le pile-ou-face.

De même, si le facteur marché a une variance 50 fois supérieure au facteur niche, une exposition égale aux deux signifierait que 98% du risque vient du marché. Pour que le facteur niche contribue autant au risque, il faut une exposition bien plus grande.

**Conséquence contre-intuitive**

Le portefeuille résultant est **sous-exposé au facteur marché** (le facteur dominant) et **sur-exposé aux facteurs de niche** (les facteurs faibles). C'est exactement l'inverse d'un portefeuille pondéré par la capitalisation (qui sur-expose au marché).

Cela signifie que le portefeuille ne "suit" pas le marché. En période de hausse généralisée, il sous-performe (car sous-exposé au facteur marché). En période de différenciation (où les facteurs de niche deviennent pertinents), il peut surperformer.

C'est le "prix" de la diversification factorielle : on renonce à suivre le facteur dominant pour être protégé contre tous les facteurs.

---

### A.4 Le seuil KL > 0.01 est-il contradictoire avec les risques dormants ?

#### L'interrogation

> Le fichier DVT ne dit pas qu'on veut aussi diversifier sur les risques dormants ? Est-ce contradictoire ? La valeur de KL correspond-elle à la confiance dans l'estimation du facteur sur tout l'historique ou la force du facteur sur une période récente ?

#### Réponse détaillée

Excellente question qui touche à une subtilité importante de l'architecture.

**Ce que mesure la divergence KL d'un facteur**

La divergence KL mesure **combien la distribution postérieure du facteur diffère du prior N(0,1)**. Pendant l'entraînement du VAE, la régularisation KL pousse chaque dimension latente vers ce prior. Seules les dimensions qui "résistent" (parce qu'elles capturent de la variance utile pour la reconstruction) restent éloignées du prior.

Un facteur avec KL > 0.01 nats est un facteur qui **encode de l'information structurelle persistante** sur l'ensemble des données d'entraînement. Ce n'est pas une mesure de la "force récente" du facteur, mais de sa **pertinence statistique cumulée** sur tout l'historique.

**Pourquoi ce n'est PAS contradictoire avec les risques dormants**

La confusion vient d'une interprétation erronée de "dormant". Un facteur dormant n'est pas un facteur qui n'existe pas — c'est un facteur qui existe mais dont la **variance actuelle est faible**.

Prenons l'exemple du facteur "contagion interbancaire" :
- En 2005-2007 : Ce facteur a une faible variance (les banques ne chutent pas ensemble particulièrement)
- En 2008 : Ce facteur explose (toutes les banques chutent simultanément)
- En 2010-2019 : Ce facteur redevient dormant (faible variance)

Pendant l'entraînement du VAE sur 1995-2019 :
- Le VAE voit les fenêtres de 2008 où ce facteur explique énormément de variance
- La pondération gamma = 3.0 amplifie ces fenêtres de crise
- Le facteur "contagion interbancaire" survit à la régularisation KL parce qu'il aide à reconstruire ces fenêtres critiques
- Sa KL finale > 0.01 reflète son importance **historique**, pas son état actuel

**Le facteur est actif (KL > 0.01) même quand il est dormant (variance actuelle faible)**

Après entraînement, la matrice d'exposition B capture les sensibilités de chaque action à ce facteur. Une banque a toujours une exposition élevée au facteur contagion, même si ce facteur ne "bouge" pas actuellement.

Lors de la construction du portefeuille, l'entropie factorielle inclut ce facteur dormant dans le calcul. Sa contribution au risque actuel est faible (car sa variance λ_k est faible), mais il est toujours là. Si sa variance augmente soudainement (nouvelle crise), les expositions sont déjà connues et le portefeuille est déjà diversifié contre lui.

**Tous les facteurs ont-ils le même "niveau de confiance" ?**

Non, et c'est normal. La KL d'un facteur mesure sa distance au prior, pas une "confiance" au sens statistique classique.

- Un facteur avec KL = 5.0 nats est très éloigné du prior (encode beaucoup d'information)
- Un facteur avec KL = 0.02 nats est proche du prior (encode peu d'information, mais assez pour survivre)

Les facteurs à faible KL (proches du seuil 0.01) sont potentiellement moins fiables — ils pourraient être du bruit qui a survécu de justesse. C'est pourquoi le tableau de bord de qualité des facteurs est utile : il permet d'identifier ces facteurs "marginaux" et de surveiller leur stabilité entre folds.

**En résumé** :
- KL > 0.01 = le facteur a une pertinence statistique sur l'historique complet
- Facteur dormant = facteur dont la variance actuelle est faible mais qui peut se réactiver
- Les deux ne sont pas contradictoires : un facteur peut être statistiquement pertinent (KL > 0.01) ET actuellement dormant (λ_k faible)

---

### A.5 Les "nats" correspondent-ils à la force d'un facteur ?

#### L'interrogation

> "Le VAE peut découvrir à la fois des facteurs forts et faibles." : est-ce ça le nats ?

#### Réponse détaillée

Non, les "nats" et la "force" d'un facteur sont deux concepts différents.

**Les nats : unité de la divergence KL**

Le "nat" (natural unit of information) est l'unité de l'entropie quand on utilise le logarithme naturel (ln) au lieu du logarithme base 2. C'est l'équivalent du "bit" en base e.

KL divergence = Σ_k (1/2) × (μ_k² + exp(log_var_k) - log_var_k - 1)  nats

Le seuil KL > 0.01 nats signifie : "la distribution postérieure de ce facteur est suffisamment éloignée du prior pour qu'on considère qu'il encode de l'information utile."

**La "force" d'un facteur : la valeur propre**

La "force" d'un facteur (au sens de Giglio et al. 2021) est sa **valeur propre** λ_k dans la matrice de covariance des facteurs Σ_z.

- Facteur fort (λ_k élevé) : Grande variance, affecte le risque de beaucoup d'actions
- Facteur faible (λ_k faible) : Petite variance, peut n'affecter qu'un sous-ensemble d'actions

**La relation entre les deux**

Un facteur peut avoir une KL élevée ET une valeur propre faible, ou vice versa :

| Situation | KL | λ_k | Interprétation |
|-----------|-----|-----|----------------|
| Facteur marché | Très élevée | Très élevée | Facteur dominant, très structuré, très variable |
| Facteur sectoriel | Moyenne | Moyenne | Facteur secondaire, structuré, variance modérée |
| Facteur de crise dormant | Faible à moyenne | Faible actuellement | Structuré (survit à KL) mais dormant (faible variance) |
| Bruit | < 0.01 | Très faible | Non structuré, ignoré |

**Ce qui distingue un facteur fort d'un facteur faible** (Giglio et al. 2021) :

- **Facteur fort** : Affecte essentiellement toutes les actions avec une contribution non-négligeable. Le facteur marché est l'exemple canonique.

- **Facteur faible** : N'affecte significativement qu'un sous-ensemble d'actions. Les facteurs sectoriels, les facteurs de niche, et les facteurs de crise sont typiquement faibles.

Les nats (KL) mesurent si le facteur encode une structure persistante. La valeur propre mesure l'amplitude de variation de cette structure. Ce sont deux dimensions orthogonales.

---

### A.6 Est-il sensé d'exclure les facteurs à faible persistance ?

#### L'interrogation

> Est-ce sensé de faire ça ? Est-ce que ça a un intérêt ? N'est-ce pas déjà fait indirectement ?

#### Réponse détaillée — pragmatique et objective

**Ce que signifierait "exclure les facteurs à faible persistance"**

Si un facteur a une demi-vie de 3 jours (ses rendements changent de signe tous les 3 jours en moyenne), alors avec un rebalancement mensuel (21 jours), le portefeuille sera constamment "en retard" sur ce facteur. L'exposition calculée au début du mois sera obsolète à la fin.

L'idée serait de ne pas inclure ce facteur dans le calcul de l'entropie, pour éviter de "gaspiller" de la capacité de diversification sur un facteur inexploitable.

**Pourquoi ce n'est PAS déjà fait indirectement**

Le VAE ne discrimine pas explicitement par persistance. Un facteur qui change de signe fréquemment mais qui explique de la variance (même sur de courtes périodes) peut survivre à la régularisation KL. Le critère KL mesure la pertinence structurelle, pas la persistance temporelle.

Cependant, il y a un effet indirect partiel :
- Les facteurs très peu persistants (demi-vie < quelques jours) tendent à être plus bruités
- Ce bruit rend leur reconstruction moins fiable
- Ils contribuent moins à réduire l'erreur de reconstruction
- Ils sont donc plus susceptibles d'être "éteints" par le KL

Mais ce n'est qu'un effet collatéral, pas un mécanisme explicite.

**Arguments POUR exclure les facteurs à faible persistance**

1. Si on ne peut pas exploiter un facteur (car il change plus vite que le rebalancement), pourquoi allouer du "budget de diversification" à ce facteur ?

2. Les facteurs à faible persistance peuvent être du bruit échantillonnal plutôt que de vraies structures.

3. Se concentrer sur les facteurs exploitables pourrait améliorer le ratio de Sharpe OOS.

**Arguments CONTRE exclure les facteurs à faible persistance**

1. **Le risque existe même si on ne peut pas l'exploiter.** Un facteur à faible persistance peut quand même causer des pertes. L'ignorer dans l'entropie signifie potentiellement se concentrer sur ce facteur par inadvertance.

2. **La persistance est estimée, pas observée.** La demi-vie calculée sur l'historique peut changer. Un facteur "peu persistant" en temps normal peut devenir très persistant en crise.

3. **C'est une forme de market timing implicite.** Décider quels facteurs "comptent" en fonction de leur persistance est une prédiction sur quels facteurs seront pertinents dans le futur.

4. **Le principe de diversification stipule de ne pas faire de paris.** L'entropie maximale traite tous les facteurs également précisément pour éviter de parier sur lesquels seront importants.

**Ma recommandation pragmatique**

**Ne pas exclure les facteurs à faible persistance de l'entropie**, mais :

1. **Calculer et monitorer la persistance** comme métrique diagnostique
2. **Ajuster la fréquence de rebalancement** si la persistance moyenne est courte (rebalancer hebdomadairement au lieu de mensuellement)
3. **Signaler une alerte** si trop de facteurs ont une persistance < fréquence de rebalancement

L'entropie doit rester agnostique sur la "qualité" des facteurs. La persistance est une information utile pour le monitoring, pas pour filtrer l'optimisation.

---

### A.7 Comparaison détaillée AU (KL) vs. Bai-Ng vs. Onatski

#### L'interrogation

> Explique plus en détail les avantages et inconvénients par rapport à la méthode actuelle.

#### Réponse détaillée

Les trois méthodes répondent à la même question : "Combien de facteurs sont réels (signal) vs. bruit ?" Mais elles utilisent des approches radicalement différentes.

**Méthode actuelle : Active Units (AU) via KL > 0.01 nats**

| Aspect | Détail |
|--------|--------|
| **Principe** | Un facteur est "actif" si sa distribution postérieure s'éloigne suffisamment du prior N(0,1) |
| **Avantages** | - Intégré au VAE, pas de calcul supplémentaire<br>- Capture les non-linéarités (contrairement à PCA)<br>- Fonctionne avec tout type de données |
| **Inconvénients** | - Le seuil 0.01 est heuristique, pas justifié statistiquement<br>- Pas de p-value ou intervalle de confiance<br>- Sensible à l'architecture du VAE (capacité K, beta, warmup) |
| **Biais connu** | Tend à surestimer le nombre de facteurs si K est grand et le beta faible |

**Méthode Bai-Ng IC2 (Information Criterion)**

| Aspect | Détail |
|--------|--------|
| **Principe** | Minimiser un critère d'information qui pénalise le nombre de facteurs : IC₂(k) = ln(MSE(k)) + k × (n+T)/(nT) × ln(min(n,T)) |
| **Avantages** | - Formellement justifié (critère d'information)<br>- Fonctionne bien pour grands n et T<br>- Indépendant des données (seuil dépend seulement de n et T) |
| **Inconvénients** | - **Tend à surestimer** le nombre de facteurs en pratique<br>- Nécessite n, T → ∞ conjointement<br>- Suppose un modèle factoriel linéaire (ne capture pas les non-linéarités du VAE) |
| **Biais connu** | "Les estimateurs IC_p1 et IC_p2 de Bai et Ng surestiment le nombre de facteurs dans tous les cas et fournissent régulièrement des estimations incorrectes" (Econometric Reviews 2018) |

**Méthode Onatski (Eigenvalue Ratio Test)**

| Aspect | Détail |
|--------|--------|
| **Principe** | Examiner les écarts entre valeurs propres adjacentes. Sous l'hypothèse nulle (pas de facteur supplémentaire), les valeurs propres du bruit suivent une distribution de Tracy-Widom |
| **Avantages** | - Fournit une p-value formelle<br>- Adapté aux données (seuil dépend de la distribution des eigenvalues)<br>- Fonctionne même quand n et T sont comparables |
| **Inconvénients** | - Suppose un modèle factoriel linéaire<br>- Sensible aux outliers dans les eigenvalues<br>- Plus complexe à implémenter |
| **Biais connu** | Peut sous-estimer quand il y a des facteurs faibles (dont les eigenvalues sont proches du bruit) |

**Comparaison synthétique**

| Critère | AU (VAE) | Bai-Ng IC2 | Onatski |
|---------|----------|------------|---------|
| Capture les non-linéarités | **Oui** | Non | Non |
| Justification statistique | Heuristique | Critère d'information | Test d'hypothèse |
| Sensibilité aux hyperparamètres | Élevée | Faible | Modérée |
| Biais typique | Variable | Surestimation | Sous-estimation (facteurs faibles) |
| Effort d'implémentation | Intégré | Faible | Modéré |

**Recommandation**

Utiliser les trois méthodes de manière complémentaire :

1. **AU (KL > 0.01)** : Méthode principale car intégrée au VAE et capturant les non-linéarités
2. **Bai-Ng IC2** : Validation croisée pour détecter une éventuelle surestimation du VAE
3. **Onatski** : Validation croisée avec p-values formelles

Si AU ≈ Bai-Ng ≈ Onatski : haute confiance dans le compte de facteurs
Si AU >> Bai-Ng ou Onatski : le VAE pourrait capturer du bruit comme facteurs
Si AU << Onatski : le seuil 0.01 nats est peut-être trop strict

Cette triangulation évite de dépendre d'une seule méthode avec ses biais spécifiques.

---

### A.8 La pondération de crise (gamma = 3.0) est-elle utile ?

#### L'interrogation

> Si on considère que toutes les crises sont différentes, non prévisibles et peu fréquentes, alors utiliser un gamma pour pondérer les crises sert-il à quelque chose car trop compliqué à caractériser / apprendre ? Ne vaut-il pas plutôt identifier des facteurs de risque non liés aux crises mais plutôt identifier les facteurs qui différencient les actions en temps normal ?

#### Réponse détaillée — Revue de littérature approfondie

C'est la question la plus profonde et la plus importante. Elle remet en cause un postulat fondamental de la stratégie.

**Le paradoxe de la diversification en crise**

La littérature académique a documenté abondamment ce phénomène, parfois appelé "diversification disappears when you need it most" :

- **Page & Panariello (2018)** dans le Financial Analysts Journal : "L'un des problèmes les plus vexants en gestion d'investissement est que la diversification semble disparaître quand les investisseurs en ont le plus besoin."

- **Kritzman & Li (2010)** : "Les corrélations sont les plus élevées quand la diversification est la plus précieuse — mais aussi quand les investisseurs sont le moins capables de l'utiliser."

- **BIS (2008)** : "Un 'breakdown de diversification' tend à se produire quand des corrélations stables sont le plus nécessaires pour la protection du portefeuille, et l'effet de diversification qui devrait protéger un portefeuille disparaît en période de pertes de marché, précisément quand il serait le plus urgemment nécessaire."

**Pourquoi les crises cassent la diversification**

En temps normal, les actions sont influencées par leurs facteurs spécifiques (secteur, style, géographie). Les corrélations reflètent ces facteurs structurels.

En crise, un mécanisme différent domine : **la liquidité**. Quand les investisseurs paniquent, ils vendent tout ce qui est vendable, indépendamment des facteurs fondamentaux. Les correlations entre toutes les classes d'actifs convergent vers 1.

Ce n'est pas que les facteurs structurels disparaissent — c'est qu'un nouveau facteur (liquidité / panique) domine temporairement et écrase tous les autres.

**L'argument CONTRE la pondération de crise (gamma = 3.0)**

Votre intuition est perspicace : si chaque crise est unique et non prévisible, qu'apprend-on en pondérant les crises passées ?

**Argument 1 : Les crises sont idiosyncratiques**

- 1987 : Crash programmatique (portfolio insurance)
- 1998 : Crise LTCM / Russie
- 2000 : Éclatement de la bulle dot-com
- 2008 : Crise des subprimes / Lehman
- 2020 : Pandémie COVID

Chaque crise a des mécanismes différents. Les actions qui chutent ensemble en 2008 (banques) ne sont pas les mêmes qu'en 2020 (compagnies aériennes).

**Argument 2 : En crise, tout chute ensemble**

Si en crise les corrélations convergent vers 1, alors aucune diversification actionnariale ne protège. Le seul refuge est hors-équité (obligations, or, cash). Apprendre les "structures de crise" dans un univers actions-only est peut-être inutile.

**Argument 3 : Faux signal**

Pondérer les crises par gamma = 3.0 force le VAE à "surappendre" des patterns rares. Avec quelques périodes de crise sur 50 ans, le modèle peut overfitter sur du bruit spécifique à ces épisodes.

**L'argument POUR la pondération de crise**

**Argument 1 : Les structures de crise ont des éléments communs**

Malgré leurs différences, les crises partagent des mécanismes :
- Flight to quality (vente des actifs risqués)
- Contagion par la liquidité (vente forcée des actifs liquides pour couvrir les pertes sur les illiquides)
- Herding (comportement grégaire)

Ces mécanismes créent des patterns de corrélation similaires : les actifs à beta élevé chutent ensemble, les actifs à faible levier résistent mieux, etc.

**Argument 2 : Les facteurs structurels persistent en crise**

Même si les corrélations augmentent en crise, elles n'atteignent pas exactement 1.0. Les différenciations persistent, bien qu'atténuées. Une entreprise avec peu de dette survit mieux qu'une entreprise très endettée, même en crise généralisée.

Le VAE peut capturer ces différenciations résiduelles en pondérant les fenêtres de crise.

**Argument 3 : La mémoire anti-cyclique**

Sans pondération de crise, le VAE optimiserait principalement pour les temps normaux (qui représentent ~90% des données). Les facteurs de crise seraient sous-représentés ou éteints par le KL.

La pondération gamma = 3.0 compense ce déséquilibre et force le modèle à allouer de la capacité aux structures de crise.

**Ce que dit la recherche empirique récente**

**COVID-19 et diversification** (Future Business Journal, 2023) :
"La crise COVID-19 a causé une faible opportunité de diversification pour les investisseurs américains, quel que soit le régime d'investissement (conventionnel ou islamique)."

**Régimes de corrélation** (Journal of Financial Economics, 2021) :
"Les corrélations entre actifs sont contextuelles, pas constantes. Elles explosent dans les crises mais reviennent à la moyenne en période de stabilité."

**L'alternative proposée : focus sur les temps normaux**

Votre suggestion alternative mérite considération :

> "Identifier les facteurs qui différencient les actions en temps normal qui ont plus de sens et sans doute basés sur des éléments fondamentaux réels"

**Avantages de cette approche** :
- Plus de données (90% du temps est "normal")
- Facteurs plus stables et interprétables
- Moins de risque d'overfitting sur des événements rares
- Les différenciations en temps normal persistent partiellement en crise

**Inconvénients de cette approche** :
- Ignore explicitement les corrélations de crise
- Le portefeuille pourrait être exposé à des facteurs dormants non détectés
- Viole le principe de "mémoire des risques dormants" de la stratégie

**Ma conclusion pragmatique**

**La pondération de crise (gamma = 3.0) est défendable mais son bénéfice est incertain.**

La vraie question est empirique : est-ce que gamma = 3.0 améliore les performances OOS par rapport à gamma = 1.0 (pas de pondération) ?

**Recommandation** :

1. **Tester gamma = 1.0 vs. gamma = 3.0** dans le walk-forward et comparer les métriques de crise (drawdown max, rendement en période de stress)

2. **Si gamma = 3.0 n'améliore pas significativement** les métriques de crise, simplifier en utilisant gamma = 1.0

3. **Alternative structurelle** : Au lieu de pondérer les crises, ajouter explicitement des facteurs de liquidité et de stress comme features (VIX, spreads de crédit, volatilité réalisée) pour que le VAE les capture naturellement

4. **Garder en tête** : Pour une vraie protection en crise, la diversification actionnariale a ses limites. La réponse pourrait être hors-scope de cette stratégie (hedging par options, allocation multi-asset).

---

## Conclusion de l'Annexe

Ces questions de suivi ont permis de clarifier des subtilités importantes :

1. **Cardinalité** : Il n'existe pas de solution "automatique" qui détermine le nombre optimal d'actions. Les contraintes fixes restent la méthode la plus simple et interprétable.

2. **Égalisation des contributions au risque** : C'est le concept central. Il implique des expositions inversement proportionnelles à la racine carrée des variances des facteurs.

3. **KL et risques dormants** : Pas de contradiction. Un facteur peut être statistiquement pertinent (KL > 0.01) et actuellement dormant (variance faible).

4. **Persistance des facteurs** : Ne pas filtrer l'entropie, mais monitorer comme diagnostic.

5. **Validation du compte de facteurs** : Triangulation AU + Bai-Ng + Onatski pour robustesse.

6. **Pondération de crise** : Question ouverte. Tester empiriquement gamma = 1.0 vs. 3.0 avant de conclure.
