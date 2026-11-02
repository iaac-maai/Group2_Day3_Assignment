---
license: mit
library_name: sklearn
tags:
  - sklearn
  - clustering
  - unsupervised-learning
  - kmeans
  - built-environment
  - housing
  - iaac-maai
  - ai-methods-1
datasets:
  - iaac-maai/day3-group2-king-county-house-typologies
---

# King County house typologies: grouping houses by their physical characteristics

This model assigns a single-family house to one of **4 physical typologies** based on 8 characteristics of the building and its plot: living area, plot area, bedrooms, bathrooms, floors, construction grade, age and the share of the living space in a basement. It is a scikit-learn pipeline (log10 of the areas + `StandardScaler` + K-Means with k = 4) fitted on 20,555 houses sold in King County (Washington, USA, including Seattle) in 2014–2015. It is meant for a housing or planning department that wants to design different programmes (retrofit, ageing-in-place, gentle densification) for different types of house. It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full analysis):** https://github.com/iaac-maai/Group2_Day3_Assignment
- **Clustered dataset:** https://huggingface.co/datasets/iaac-maai/day3-group2-king-county-house-typologies
- **Team:** Karthikeyan Saravanan, Emmanuel Marquez, Bars Babila, Samiha Abeda Nur

## Intended use

- **Intended users and use:** planners, housing departments, researchers and students who want to describe a stock of single-family houses with a few understandable types, for example to estimate how many houses of each type a programme would reach, or to compare price, location or condition between types.
- **Out-of-scope uses:**
  - Houses outside King County, or outside the 2014–2015 market: the types reflect the local building history of this county.
  - Apartments, condominiums or non-residential buildings: the data only has houses.
  - Houses outside the ranges of the training data (for example more than 4,970 sqft of living space or more than 6 bedrooms, which were removed in our cleaning).
  - Any prediction of price, energy performance, structural safety or value: the model only says which group of physically similar houses a house is closest to.
  - Decisions about individual households: the groups describe buildings, not people.

## Model details

| | |
|---|---|
| **Task** | Clustering (unsupervised) |
| **Algorithm** | `KMeans` |
| **Key hyperparameters** | `n_clusters=4`, `n_init=10`, `random_state=42` (other parameters at scikit-learn defaults) |
| **Preprocessing** | `log10` of `sqft_living` and `sqft_lot` (`FunctionTransformer` in a `ColumnTransformer`), then `StandardScaler` on all 8 features |
| **Assigner** | Not needed: K-Means assigns new houses to the nearest centre with `.predict()` |
| **File** | `model.joblib`: the whole fitted scikit-learn `Pipeline` (steps `preprocess` and `kmeans`) |
| **Library versions** | scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, joblib 1.6.0 |
| **Authors** | Group 2, AI Methods I, IAAC MaAI 2026/27 |

## Input features

The model expects a `pandas.DataFrame` with exactly these columns, in **original units** (the pipeline does the log and the scaling itself):

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| `sqft_living` | numeric | sqft | Interior living area (above ground + basement) | log10, scaling |
| `sqft_lot` | numeric | sqft | Plot area | log10, scaling |
| `bedrooms` | integer | count | Number of bedrooms (2–6 in the data) | scaling |
| `bathrooms` | numeric | count | Number of bathrooms; 0.5 = toilet without shower, 0.75 = with shower but no bath | scaling |
| `floors` | numeric | count | Number of storeys (1–3.5; .5 = a partial storey) | scaling |
| `grade` | integer | 1–13 | King County construction and design grade (4–13 in the data; 7 = average) | scaling |
| `house_age` | integer | years | Age of the house at the time of sale: sale year − year built (0 if sold before completion) | scaling |
| `basement_share` | numeric | 0–1 | `sqft_basement / sqft_living` (0 = no basement) | scaling |

## Output and clusters

`model.predict(X)` returns the cluster number of each row (0, 1, 2 or 3):

| Cluster | Name | Share of the data | Defining features (medians) |
|---|---|---|---|
| 0 | Large Modern Family Homes | 23.0% (4,728) | 2,960 sqft, 4 bedrooms, 2.5 bathrooms, grade 9, 2 storeys, built around 1995 |
| 1 | Postwar Single-Storey Homes | 31.5% (6,467) | 1,320 sqft, 3 bedrooms, 1 bathroom, grade 7, 1 storey (79%), built around 1953 |
| 2 | Contemporary Compact Homes | 19.2% (3,940) | 1,790 sqft on a 4,084 sqft plot, 2 storeys (13% have 3), grade 8, built around 2004 |
| 3 | Mid-Century Basement Homes | 26.4% (5,420) | 2,130 sqft, 97% with a basement (median share 0.37), 1 storey (84%), built around 1962 |

![Cluster profiles](https://huggingface.co/datasets/iaac-maai/day3-group2-king-county-house-typologies/resolve/main/09_cluster_profiles.png)

The names are the team's interpretation. They were checked against variables the model never saw: the median price goes from about \$350,000 (cluster 1) to \$695,000 (cluster 0); clusters 1 and 3 are closer to downtown Seattle (about 15 km) than cluster 0 (21 km); and about half of clusters 1 and 3 have a condition rating of 4 or 5, against 8% in cluster 2.

## Training data

| | |
|---|---|
| **Dataset** | House Sales in King County, USA |
| **Source** | [original (Kaggle)](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) · [our clustered version](https://huggingface.co/datasets/iaac-maai/day3-group2-king-county-house-typologies) |
| **Licence** | Original: CC0 (public domain). Our clustered version: CC BY 4.0 |
| **One row is** | One house |
| **Size used** | 20,555 houses |
| **Coverage** | King County, Washington (USA), houses sold between May 2014 and May 2015 |

**Cleaning and feature engineering:** the Day 1 and Day 2 cleaning removed impossible and extreme values (prices, areas, bedroom counts) and the `waterfront` and `view` columns, leaving 20,703 sales. 148 houses were sold twice without any physical change; we kept only the latest sale, so one row is one house. We engineered `house_age`, `basement_share` (both used by the model) and other features used only for interpretation (`dist_downtown_km`, `price_per_sqft`, `lot_coverage`, `living_vs_neighbours`, `renovated`, `years_since_work`, `sale_year`, `sale_month`). Price and location were deliberately **not** used for clustering.

## Method and evaluation

- **Methods compared:** K-Means on all houses and Ward hierarchical clustering on a fixed sample of 2,000 houses (Ward does not scale to 20,555 rows). Agreement on the same 2,000 houses: **ARI = 0.580** (moderate). Ward gave one small group (128 of 2,000 houses).
- **Choice of k:** we tested k = 2 to 9. The inertia decreases without a clear elbow; the silhouette (on 2,000 houses) is 0.278 at k = 2, 0.275 at k = 3 and **0.264 at k = 4**. No k stands out, so we chose k = 4 for interpretability: with k = 3, the large modern homes and the compact new-builds are merged, and with k = 2 the groups are only "old" and "new".
- **Stability:** ARI against the final model of **0.991–1.000** over five random seeds, and **0.976–0.997** for K-Means refitted on five random 80% subsamples (compared on the shared rows).
- **Dimensionality reduction (context):** PCA shows about 3–4 real dimensions (four components explain 84.2% of the variance). PC1 is "how much house" and PC2 the building form, and the four clusters are four slices of one continuous cloud along these two axes.
- **Why this model:** K-Means is very stable on this data, uses every house, gives groups of reasonable size (3,940 to 6,467 houses) that we could name, and can assign new houses directly with `.predict()`.

## Limitations and biases

- **Weak separation:** the silhouette is low (0.264). The data is a continuum, and the groups are useful cuts of it, not natural types. A house near a border can easily be assigned to the neighbouring group.
- **Sensitive to our choices:** other features (for example adding price or location), another preprocessing or another k would give different groups. The stability tests only change the random seed and the sample.
- **Cluster 2 combines two contexts:** urban townhouse infill (about a third of the group, plots under 2,500 sqft, about 8 km from downtown) and new suburban subdivisions (about 25 km). They are physically similar but may need different policies.
- **Data bias:** only houses sold in one year are included, so the stock is sampled, not complete. The most extreme houses were removed in cleaning. `grade` is an assessor's score, not a measurement.
- **Place and time:** the types reflect King County's building history; the model should not be used for other regions.

## How to use

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/day3-group2-house-typologies", filename="model.joblib")
model = joblib.load(path)

CLUSTER_NAMES = {
    0: "Large Modern Family Homes",
    1: "Postwar Single-Storey Homes",
    2: "Contemporary Compact Homes",
    3: "Mid-Century Basement Homes",
}

example = pd.DataFrame([{
    "sqft_living": 1500,     # sqft
    "sqft_lot": 1300,        # sqft
    "bedrooms": 3,
    "bathrooms": 2.5,
    "floors": 3.0,
    "grade": 8,              # King County grade, 1-13
    "house_age": 1,          # years
    "basement_share": 0.0,   # sqft_basement / sqft_living
}])
cluster = int(model.predict(example)[0])
print(cluster, CLUSTER_NAMES[cluster])   # 2 Contemporary Compact Homes
```

⚠️ `.joblib` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above (especially scikit-learn 1.9.1); the model may not load with other versions.

## Reproducibility

The full analysis (data loading, cleaning check, feature engineering, PCA, t-SNE, UMAP, clustering, stability, interpretation and export) is in the GitHub repository: https://github.com/iaac-maai/Group2_Day3_Assignment. Run `notebook/unsupervised_learning.ipynb` with the packages in `requirements.txt`. A fixed `random_state=42` is used everywhere.

## Authors and licence

Karthikeyan Saravanan, Emmanuel Marquez, Bars Babila, Samiha Abeda Nur. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The training data is based on *House Sales in King County, USA* (Kaggle, CC0 public domain); our clustered version is released under CC BY 4.0.
