# Discovering Movie Audience Segments and Hidden Viewing Preferences

DSAI 4102 – Applied Data Mining, Fall 2026, UDST

Group: Mouaz Djebbar, Mohid Intikhab, Abdallah Dahmani

## Dataset
MovieLens 32M from GroupLens: https://grouplens.org/datasets/movielens/32m/
Download `ml-32m.zip`, unzip it, and put `ratings.csv` and `movies.csv` in the `data/` folder.
The cleaned files (`sample_ratings.parquet`, `clean_ratings.parquet`, etc.) are shared on Google Drive.
Put them in `data/` too. The `data/` folder is not uploaded to GitHub because the files are too big.

## Setup
```
pip install -r requirements.txt
```

## Notebooks (run in this order)
| Notebook | Owner | Content |
|---|---|---|
| 01_data_eda.ipynb | Mouaz | Loading, cleaning, EDA |
| 02_clustering.ipynb | Mouaz | User profiles and K-Means |
| 03_rules.ipynb | Mohid | Transactions, Apriori, FP-Growth |
| 04_pca_robustness.ipynb | Abdallah | PCA and stability checks |
| 05_cluster_rules.ipynb | Mouaz | Rules compared across clusters |

## Shared rules
- Liked = rating >= 4.0
- IMAX and "(no genres listed)" are dropped, leaving 18 genres
- Random seed = 42
