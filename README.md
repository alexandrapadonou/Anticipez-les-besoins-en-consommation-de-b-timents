# Anticipation des consommations d'énergie et des émissions de CO₂ des bâtiments de Seattle

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-latest-2EA44F)
![LIME](https://img.shields.io/badge/LIME-latest-FF6F00)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

Projet d'analyse et de modélisation visant à prédire la **consommation énergétique** et les **émissions de gaz à effet de serre (GES)** des bâtiments **non résidentiels** de la ville de Seattle, à partir de leurs caractéristiques structurelles (type, surface, nombre d'étages, année de construction, quartier, etc.).

L'objectif est d'éviter à la ville d'avoir à réaliser des relevés coûteux et chronophages, en estimant ces valeurs directement à partir des données déclaratives des bâtiments.

## Données

- **Source** : [2016 Building Energy Benchmarking – City of Seattle](https://data.seattle.gov/Built-Environment/Building-Energy-Benchmarking-Data-2015-Present/teqw-tu6e/about_data)
- **Fichier brut** : [2016_Building_Energy_Benchmarking.csv](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P4/2016_Building_Energy_Benchmarking.csv)
- Chaque ligne correspond à un bâtiment / une propriété et décrit son identification, ses caractéristiques physiques, sa consommation d'énergie, sa performance énergétique et ses émissions de GES.

### Cibles prédites

| Cible | Description | Notebook |
|-------|-------------|----------|
| `SiteEnergyUseWN(kBtu)` | Consommation totale d'énergie du site (normalisée météo) | Notebook 2 |
| `GHGEmissionsIntensity` | Intensité des émissions de gaz à effet de serre | Notebook 3 |

## Structure du projet

```

├── Padonou_Alexandra_1_notebook_exploratoire_012025.ipynb   # Analyse exploratoire & nettoyage
├── Padonou_Alexandra_2_notebook_prediction_012025.ipynb     # Prédiction de la consommation d'énergie
├── Padonou_Alexandra_3_notebook_prediction_012025.ipynb     # Prédiction de l'intensité des émissions de GES
├── 2016_Building_Energy_Benchmarking.csv                    # Jeu de données brut (source)
├── AED_building.csv                                         # Jeu nettoyé produit par le notebook 1
└── README.md
```

> Les fichiers `building.csv` et `bâtiments_non_résidentiels.csv` sont des exports intermédiaires/optionnels du travail.

## Notebook 1 - Analyse exploratoire des données (AED)

Notebook : `Padonou_Alexandra_1_notebook_exploratoire_012025.ipynb`

Objectifs : comprendre les données, identifier les tendances et corriger les problèmes de qualité.

Étapes principales :
- **Filtrage** : conservation des seuls bâtiments **non résidentiels** et des propriétés au statut `Compliant`.
- **Compréhension et choix des features** : suppression des colonnes inutiles (identifiants, adresse, coordonnées, commentaires, colonnes d'énergie redondantes, etc.).
- **Gestion des valeurs manquantes** : suppression des lignes sans information clé, imputation par la **médiane par type de propriété** (`PrimaryPropertyType`).
- **Traitement des outliers** :
  - `NumberofBuildings` à 0 reconstruit à partir de `ListOfAllPropertyUseTypes`.
  - `NumberofFloors` à 0 imputé par la médiane.
  - Filtrage / nettoyage des valeurs aberrantes ou négatives sur `Electricity(kBtu)`, `GHGEmissionsIntensity`, `SiteEnergyUseWN(kBtu)`, `TotalGHGEmissions`, `SteamUse(kBtu)`.
  - Correction des cas où `LargestPropertyUseTypeGFA` > `PropertyGFATotal`.
- **Analyse univariée / bivariée** : distributions, normalisation des noms de quartiers, matrice de corrélation.
- **Variables cibles** : mise en évidence d'une distribution asymétrique et vérification de l'effet d'une **transformation logarithmique** (`log1p`).
- **Sortie** : export du jeu nettoyé dans `AED_building.csv`.

## Notebooks 2 & 3 - Modélisation

Les deux notebooks suivent la même démarche, mais ciblent une variable différente :
- **Notebook 2** → prédiction de `SiteEnergyUseWN(kBtu)` (consommation d'énergie).
- **Notebook 3** → prédiction de `GHGEmissionsIntensity` (intensité des émissions de GES).

### Feature engineering

- **`YearBuilt`** transformée en catégories (`YearBuiltRange`), avec un découpage en deux classes dans l'itération finale (avant / après une année charnière).
- **`NumberofFloors`** transformée en tranches (`NumberofFloorsRange`).
- Regroupement possible des types de propriété en catégories plus générales (`GeneralPropertyType`).
- Création de **proportions** : `ProportionParking`, `ProportionBuilding`, et proportions énergétiques (`ProportionOfSteamUse`, `ProportionOfElectricity`, `ProportionOfNaturalGas`).
- **Encodage** des variables catégorielles : One-Hot Encoding (années, quartiers, types) et Label Encoding (tranches d'étages).
- Vérification de la **corrélation** entre nouvelles variables.
- Système d'**itérations** (paramètre `iteration`) permettant de tester différentes stratégies de feature engineering ; l'itération 6 est retenue.

### Préparation pour l'entraînement

- Séparation `X` / `y`, split train/test (80 / 20, `random_state=42`).
- **Transformation logarithmique** de la cible (`log1p`) et inversion (`expm1`) au moment de l'évaluation.
- **Standardisation** des variables explicatives (`StandardScaler`).

### Modèles testés

| Modèle | Rôle |
|--------|------|
| `DummyRegressor` (moyenne) | Modèle de référence (baseline) |
| `Ridge` | Régression linéaire régularisée (L2) |
| `ElasticNet` | Régression linéaire régularisée (L1 + L2) |
| `DecisionTreeRegressor` | Arbre de décision |
| `GradientBoostingRegressor` | Modèle d'ensemble le plus performant |

- Recherche d'hyperparamètres via **`GridSearchCV`** (validation croisée 5 folds).
- **Métriques** d'évaluation : R², RMSE, MAE, ainsi que le temps d'entraînement et de prédiction.
- Visualisations : prédictions vs valeurs réelles, **courbes d'apprentissage** (détection sur-/sous-apprentissage).

### Interprétabilité

- **SHAP** : importance globale et locale des variables.
- **LIME** : explication locale de prédictions individuelles.

### Évaluation de l'ENERGY STAR Score

Chaque notebook compare deux modèles afin de mesurer l'apport de la variable `ENERGYSTARScore` :
- un modèle **avec** `ENERGYSTARScore` ;
- un modèle **sans** `ENERGYSTARScore`.

Cette comparaison permet de déterminer si cette note (coûteuse à obtenir) améliore réellement la qualité des prédictions.

## Prérequis et installation

Environnement Python 3 avec les bibliothèques suivantes :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap lime jupyter
```

## Utilisation

Exécuter les notebooks **dans l'ordre** :

1. `Padonou_Alexandra_1_notebook_exploratoire_012025.ipynb` - génère `AED_building.csv`.
2. `Padonou_Alexandra_2_notebook_prediction_012025.ipynb` - prédiction de la consommation d'énergie.
3. `Padonou_Alexandra_3_notebook_prediction_012025.ipynb` - prédiction de l'intensité des émissions de GES.

```bash
jupyter notebook
```

> Les notebooks 2 et 3 s'appuient sur le fichier `AED_building.csv` produit par le notebook 1 : il doit donc être généré au préalable.

## Auteur

Alexandra Padonou - Projet OpenClassrooms.
