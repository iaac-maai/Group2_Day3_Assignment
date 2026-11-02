---
license: cc-by-4.0
pretty_name: King County house typologies (2014–2015), with physical clusters
size_categories:
  - 10K<n<100K
tags:
  - built-environment
  - housing
  - real-estate
  - clustering
  - unsupervised-learning
  - iaac-maai
  - ai-methods-1
configs:
  - config_name: default
    data_files:
      - split: train
        path: king_county_house_typologies.csv
---

# King County house typologies (2014–2015), with physical clusters

**20,555 single-family houses** sold in King County (Washington, USA, which includes Seattle) between May 2014 and May 2015. Each row is **one house**: its sale (date, price), its physical characteristics, its approximate location, ten engineered features, and the **physical typology** it belongs to ("Large Modern Family Homes", "Postwar Single-Storey Homes", "Contemporary Compact Homes", "Mid-Century Basement Homes"), found by clustering the building characteristics only. Each house also has its position on three 2D maps (PCA, t-SNE and UMAP), so the groups can be plotted without recomputing anything.

The dataset is meant for anyone who wants to study a housing stock by **type of building**, for example a planning department that needs to know which kinds of house a retrofit or densification programme would reach.

- **GitHub repository (how it was made):** https://github.com/iaac-maai/Group2_Day3_Assignment (notebook `notebook/unsupervised_learning.ipynb`)
- **Model fitted on it:** https://huggingface.co/iaac-maai/day3-group2-house-typologies
- **Team:** Karthikeyan Saravanan, Emmanuel Marquez, Bars Babila, Samiha Abeda Nur

## Source and attribution

| | |
|---|---|
| **Original dataset** | [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) on Kaggle (uploaded by harlfoxem), based on public King County sales records |
| **Original licence** | CC0 1.0 (public domain) |
| **Original content** | 21,613 house sales with price, date, 18 house and location columns |

This dataset is a derived work. The original is in the public domain; we release our version under CC BY 4.0.

## What we changed

- **Cleaning (Day 1 and Day 2 of the course):** removed impossible and extreme values (prices, living areas, bedroom counts) and the `waterfront` and `view` columns: 21,613 → **20,703 sales**. No missing values remain.
- **One row per house:** 148 houses were sold twice in the year. None of them changed physically between the two sales, so we kept only the latest sale: 20,703 → **20,555 houses**. For those houses, `date` and `price` refer to the latest sale.
- **Privacy:** we **removed the original `id`** (the county parcel number, which links to the public property register and its owner) and replaced it with our own `house_id`. We **rounded `lat` and `long` to 3 decimals** (about 100 m). `dist_downtown_km` was computed before rounding.
- **Feature engineering (10 new columns):** `sale_year`, `sale_month`, `house_age`, `renovated`, `years_since_work`, `basement_share`, `lot_coverage`, `living_vs_neighbours`, `price_per_sqft`, `dist_downtown_km` (see the column table).
- **Clustering:** the `cluster` and `cluster_name` columns come from the model above: K-Means (k = 4) on 8 physical features (`sqft_living`, `sqft_lot`, `bedrooms`, `bathrooms`, `floors`, `grade`, `house_age`, `basement_share`), after a log10 of the two areas and standard scaling. **Price and location were not used**, so they can be used to check the groups. The coordinates come from PCA, t-SNE and UMAP of the same 8 scaled features.

## The clusters

![The four typologies on the PCA, t-SNE and UMAP maps](https://huggingface.co/datasets/iaac-maai/day3-group2-king-county-house-typologies/resolve/main/11_named_clusters_on_maps.png)

| Cluster | Name | Share of the rows | What makes it different (medians) |
|---|---|---|---|
| 0 | Large Modern Family Homes | 23.0% (4,728) | The biggest and best-built houses: 2,960 sqft, 2.5 bathrooms, grade 9, 2 storeys, built around 1995. Most expensive (\$695k) and furthest out (21 km, Eastside suburbs such as Sammamish and Redmond) |
| 1 | Postwar Single-Storey Homes | 31.5% (6,467) | The smallest and oldest: 1,320 sqft, 1 bathroom, grade 7, 79% single-storey, built around 1953. Cheapest (\$350k) but the highest price per sqft (\$270), mostly inner north Seattle |
| 2 | Contemporary Compact Homes | 19.2% (3,940) | The newest (built around 2004) on the smallest plots (4,084 sqft), 2–3 storeys. A third are urban townhouses (plots under 2,500 sqft, about 8 km from downtown), two thirds new suburban houses (about 25 km) |
| 3 | Mid-Century Basement Homes | 26.4% (5,420) | 97% have a basement, with a third of the living space below ground; 84% single-storey, 4 bedrooms, built around 1962. Median \$495k, 14 km from downtown, the best rated condition |

![Cluster profiles](https://huggingface.co/datasets/iaac-maai/day3-group2-king-county-house-typologies/resolve/main/09_cluster_profiles.png)

**How well separated are they?** Not very: the silhouette score is about 0.26, and on the PCA map the four groups are **neighbouring slices of one continuous cloud**. The first PCA axis (42% of the variance) is "how much house" and separates cluster 0 from cluster 1; the second (21%) is the building form and separates the basement houses (cluster 3) from the multi-storey new-builds (cluster 2). The groups are a useful way of cutting a continuum, not separate islands. They are, however, very **stable**: re-running K-Means with other seeds or on 80% subsamples gives an adjusted Rand index of 0.98–1.00. On the t-SNE and UMAP maps each group is made of many small islands: these are combinations of discrete values (bedrooms, floors, grade, basement), not sub-types.

**Variables that were not used** also differ between the groups: price (median \$350k to \$695k), distance to downtown (14 to 21 km) and condition (about half of clusters 1 and 3 are rated 4 or 5, against 8% of cluster 2, since condition is rated relative to the age of the house). Price per square foot, on the other hand, is almost the same in every group (\$233–270): it depends on location more than on the type of building.

## Columns

| Column | Type | Unit | Description |
|---|---|---|---|
| `house_id` | integer | – | Our identifier of the house (1 … 20,555); replaces the original parcel `id` |
| `date` | date | – | Date of the (latest) sale, YYYY-MM-DD |
| `price` | float | USD | Sale price |
| `bedrooms` | integer | count | Number of bedrooms |
| `bathrooms` | float | count | Number of bathrooms (0.5 = toilet only, 0.75 = shower but no bath) |
| `sqft_living` | integer | sqft | Interior living area |
| `sqft_lot` | integer | sqft | Plot area |
| `floors` | float | count | Number of storeys (.5 = a partial storey) |
| `condition` | integer | 1–5 | Condition rated by the county assessor, relative to the age of the house |
| `grade` | integer | 1–13 | County construction and design grade (7 = average) |
| `sqft_above` | integer | sqft | Living area above ground |
| `sqft_basement` | integer | sqft | Living area in the basement (`sqft_above + sqft_basement = sqft_living`) |
| `yr_built` | integer | year | Year built |
| `yr_renovated` | integer | year | Year of the last renovation (0 = never renovated) |
| `zipcode` | integer | – | ZIP code (70 in the data) |
| `lat`, `long` | float | degrees | Location, **rounded to 3 decimals** (about 100 m) |
| `sqft_living15` | integer | sqft | Average living area of the 15 nearest houses |
| `sqft_lot15` | integer | sqft | Average plot area of the 15 nearest houses |
| `sale_year`, `sale_month` | integer | – | Year and month of the sale |
| `house_age` | integer | years | `sale_year − yr_built`, set to 0 for the 11 houses sold before completion. **Used for clustering** |
| `renovated` | integer | 0/1 | 1 if the house was ever renovated |
| `years_since_work` | integer | years | Years since the latest of construction and renovation |
| `basement_share` | float | 0–1 | `sqft_basement / sqft_living`. **Used for clustering** |
| `lot_coverage` | float | ratio | Approximate footprint over plot: `(sqft_above / floors) / sqft_lot` (a rough estimate; 4 rows exceed 1) |
| `living_vs_neighbours` | float | ratio | `sqft_living / sqft_living15`: is the house big for its street? |
| `price_per_sqft` | float | USD/sqft | `price / sqft_living` |
| `dist_downtown_km` | float | km | Great-circle distance to downtown Seattle (47.6062, −122.3321) |
| `cluster` | integer | – | Typology assigned by the model (0–3) |
| `cluster_name` | text | – | Name of the typology (our interpretation, see the table above) |
| `pca_x`, `pca_y` | float | – | Position on the first two principal components of the 8 scaled clustering features (63.2% of their variance). A linear projection: its axes can be read |
| `tsne_x`, `tsne_y` | float | – | Position on a 2D t-SNE map (`perplexity=30`, `init="pca"`, `random_state=42`). Only for plotting: sizes of and distances between groups are not meaningful |
| `umap_x`, `umap_y` | float | – | Position on a 2D UMAP map (`n_neighbors=30`, `min_dist=0.1`, `random_state=42`). Only for plotting, with the same caveat as t-SNE |

## Limitations and biases

- **Place and time:** one county, one year (2014–2015). Prices and the stock have changed since, and the types reflect this county's building history.
- **Sold houses only:** houses that were not sold in that year are missing, so this is a sample of the stock, not a census. Apartments and condominiums are not included.
- **Inherited cleaning:** the most extreme houses (very large, very expensive, waterfront) were removed on Days 1 and 2, so unusual houses are under-represented.
- **Coded variables:** `grade` and `condition` are assessor scores, not measurements. `lot_coverage` is approximate.
- **The clusters are one possible reading:** the data is a continuum (silhouette ≈ 0.26), so another algorithm, other features or another k would draw other borders. Ward hierarchical clustering agreed only moderately with K-Means (ARI 0.58 on 2,000 houses). Cluster 2 groups urban townhouses with new suburban houses. The names are our interpretation of the buildings, not information about the people living in them.
- **Not reproducible exactly:** the t-SNE and UMAP coordinates can differ on another computer or library version.

## How to load it

```python
import pandas as pd

df = pd.read_csv(
    "hf://datasets/iaac-maai/day3-group2-king-county-house-typologies/king_county_house_typologies.csv",
    parse_dates=["date"],
)
print(df.groupby("cluster_name")[["price", "sqft_living", "yr_built"]].median())

# The typologies on a map: pca_x/pca_y, tsne_x/tsne_y or umap_x/umap_y
import matplotlib.pyplot as plt

for name, rows in df.groupby("cluster_name"):
    plt.scatter(rows["pca_x"], rows["pca_y"], s=3, label=name)
plt.legend(markerscale=4)
plt.show()

# ... or on the real map of the county
for name, rows in df.groupby("cluster_name"):
    plt.scatter(rows["long"], rows["lat"], s=2, label=name)
plt.legend(markerscale=4)
plt.show()
```

Reading `hf://` paths requires the `huggingface_hub` package.

## Related

- **GitHub repository (full analysis):** https://github.com/iaac-maai/Group2_Day3_Assignment
- **Model (assigns a new house to a typology):** https://huggingface.co/iaac-maai/day3-group2-house-typologies

## Authors and licence

Karthikeyan Saravanan, Emmanuel Marquez, Bars Babila, Samiha Abeda Nur. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Released under **CC BY 4.0**, based on *House Sales in King County, USA* (Kaggle, CC0 1.0 public domain).
