# Implémentations d'Algorithmes d'Apprentissage Automatique

![MindSpore Framework](https://upload.wikimedia.org/wikipedia/commons/3/30/MindSpore-logo.png) <!-- Remplacez cette URL par une image réelle hébergée -->
- **Mindspore** : Information sur l'installation depuis [https://www.mindspore.cn/install/en].

![Version Python](https://img.shields.io/badge/Python-3.x-bleu)

Bienvenue dans le dépôt **Implémentations d'Algorithmes d'Apprentissage Automatique** ! Cette collection de notebooks Jupyter propose des implémentations concrètes d'algorithmes clés d'apprentissage automatique. Chaque fichier exécute des tâches spécifiques — construction, entraînement et visualisation de modèles — en utilisant des datasets ou des données générées. Que vous soyez débutant ou passionné, explorez les détails ci-dessous pour découvrir ces techniques fondamentales, avec des exemples compatibles avec le framework **MindSpore** !

---

## Vue d'Ensemble des Fichiers

### 1. [LogisticRegression.ipynb](LogisticRegression.ipynb)
- **Ce que fait le code** :
  - Importe `StandardScaler` et `LogisticRegression` de `scikit-learn` pour la standardisation et la classification binaire.
  - Définit un dataset personnalisé avec des paires (loyer, superficie) et des étiquettes binaires (0 : non, 1 : oui).
  - Standardise les données pour normaliser les variances (moyenne = 0, variance = 1) afin d'éviter la domination de certaines features.
  - Entraîne un modèle de régression logistique et prédit sur un nouvel exemple (ex. : [2000, 8]) avec des probabilités (ex. : [0.4189, 0.5811] pour non/oui).
- **Objectif** : Construit un modèle de classification binaire pour prédire des décisions (ex. : louer ou non) à partir de features numériques.
- **Dépendances** : `scikit-learn`, `numpy`. Compatible avec MindSpore pour des implémentations avancées.
- **Note** : Peut être adapté pour utiliser les capacités de MindSpore pour l'entraînement distribué.

### 2. [LinearRegressionImplementation.ipynb](LinearRegressionImplementation.ipynb)
- **Ce que fait le code** :
  - Importe `numpy` et `matplotlib.pyplot` pour les calculs et les visualisations.
  - Charge un dataset simulé (`Ir2_data.txt`) avec des paires (superficie, prix).
  - Implémente des fonctions pour :
    - Calculer les gradients d'une régression linéaire.
    - Initialiser les paramètres (theta) avec des uns.
    - Réaliser une descente de gradient pour ajuster le modèle.
    - Visualiser les courbes de perte et les lignes de régression ajustées.
  - Entraîne le modèle jusqu'à ce que le gradient soit inférieur à 0.00001, en enregistrant la perte tous les 10 pas.
  - Affiche les résultats visuels des données et de l'ajustement.
- **Objectif** : Construit une régression linéaire de zéro avec descente de gradient, avec retour visuel.
- **Dépendances** : `numpy`, `matplotlib`. Peut être porté vers MindSpore pour une optimisation avancée.
- **Note** : Idéal pour comparer avec les optimisations de MindSpore.

### 3. [LinearRegression.ipynb](LinearRegression.ipynb)
- **Ce que fait le code** :
  - Importe `LinearRegression` de `scikit-learn`, ainsi que `numpy` et `matplotlib.pyplot` pour les calculs et les graphiques.
  - Crée un dataset simple avec des paires (superficie, prix) (ex. : [121, 300], [161, 517]).
  - Entraîne un modèle de régression linéaire avec `fit`.
  - Extrait et affiche les paramètres du modèle (ex. : pente = 4.98, ordonnée = -274.88).
  - Visualise les données avec un nuage de points et trace la droite de régression.
  - Prédit une valeur pour une nouvelle entrée (ex. : 130 → prix prédit = 373.13).
- **Objectif** : Construit et visualise un modèle de régression linéaire simple pour prédire des valeurs continues.
- **Dépendances** : `scikit-learn`, `numpy`, `matplotlib`. Compatible avec MindSpore pour des déploiements scalables.
- **Note** : Peut être enrichi avec les outils de MindSpore pour des performances accrues.

### 4. [K-meansAlgorithmImplementation.ipynb](K-meansAlgorithmImplementation.ipynb)
- **Ce que fait le code** :
  - Importe `make_blobs`, `matplotlib.pyplot`, et `KMeans` de `scikit-learn` pour générer des données et effectuer le clustering.
  - Génère un dataset synthétique de 500 échantillons avec 2 features et 4 centres via `make_blobs`.
  - Affiche les dimensions du dataset (500, 2 pour X, 500 pour y).
  - Crée des graphiques de dispersion :
    - Un premier sans étiquettes pour montrer les clusters.
    - Un second avec des couleurs basées sur les étiquettes générées.
  - Applique le clustering K-means pour 3 et 4 clusters, en affichant les centroïdes.
  - Visualise les résultats avec des marqueurs pour les centroïdes et des couleurs pour les clusters.
- **Objectif** : Implémente et visualise le clustering K-means sur un dataset généré avec différents nombres de clusters.
- **Dépendances** : `scikit-learn`, `matplotlib`. Peut être optimisé avec MindSpore pour des datasets plus grands.
- **Note** : Convient pour une transition vers les capacités de clustering de MindSpore.

### 5. [DecisionTree.ipynb](DecisionTree.ipynb)
- **Ce que fait le code** :
  - Importe `pandas`, `numpy`, `tree` de `scikit-learn`, et `pydotplus` pour la manipulation de données et la visualisation.
  - Charge un dataset (`tennis.txt`) avec 14 échantillons (ex. : conditions météo) et une étiquette binaire (oui/non).
  - Convertit les données catégoriques en valeurs numériques avec `pd.Categorical`.
  - Entraîne un arbre de décision avec l'entropie comme critère de séparation.
  - Exporte et sauvegarde un diagramme de l'arbre en PDF (`tennis.pdf` si généré).
  - Prédit une nouvelle entrée (ex. : [0, 0, 1, 1] → "N").
- **Objectif** : Construit un arbre de décision pour la classification et génère une représentation visuelle.
- **Dépendances** : `pandas`, `scikit-learn`, `pydotplus`, `matplotlib`. Compatible avec MindSpore pour des arbres plus complexes.
- **Note** : Peut être adapté pour utiliser les outils d'apprentissage profond de MindSpore.

### 6. [ML](ML/)
- **Ce que fait le code** : Pas un script exécutable, mais un dossier contenant les datasets utilisés par les notebooks (ex. : `tennis.txt` pour `DecisionTree.ipynb`, `Ir2_data.txt` pour `LinearRegressionImplementation.ipynb`).
- **Objectif** : Fournit les fichiers de données brutes nécessaires aux calculs et visualisations des notebooks.
- **Dépendances** : Aucun (fichiers de données uniquement).

---

## Prérequis
- **Python 3.x** : [Téléchargez depuis python.org](https://www.python.org/downloads/).
- **Dépendances** : Installez les bibliothèques requises avec :
  ```bash
  pip install scikit-learn numpy matplotlib pandas pydotplus mindspore
