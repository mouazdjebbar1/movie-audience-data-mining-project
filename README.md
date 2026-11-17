# Discovering Movie Audience Segments and Hidden Viewing Preferences

DSAI 4102 – Applied Data Mining, Fall 2026, UDST

Group: Mouaz Djebbar, Mohid Intikhab, Abdallah Dahmani

## Dataset
MovieLens 32M from GroupLens: https://grouplens.org/datasets/movielens/32m/
Download `ml-32m.zip`, unzip it, and put `ratings.csv` and `movies.csv` in the `data/` folder.
Then run `notebooks/01_data_eda.ipynb` once. It creates the cleaned files (`sample_ratings.parquet`, `clean_ratings.parquet`, `movies_clean.parquet`) in `data/`.
The `data/` folder is not uploaded to GitHub because the files are too big.

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

## Cleaning steps (done in 01_data_eda)
1. **Checked the files** against the official GroupLens MD5 checksums. `ratings.csv` and `movies.csv` both match, so our data is the official MovieLens 32M.
2. **Checked data quality.** No missing values and no duplicate user-movie ratings, so nothing needed fixing.
3. **Removed the IMAX label.** IMAX is a screen format, not a genre. The movies stay with their other genres (for example, `Action|Sci-Fi|IMAX` becomes `Action|Sci-Fi`).
4. **Removed movies with no genres.** 7,081 movies had no genres left (7,080 with "(no genres listed)" and 1 that only had IMAX). This removed only 55,528 ratings (0.17%), and no users were lost.
5. **Kept all users**, including a few who now have 18 or 19 ratings instead of 20. The 20 was GroupLens's rule for selecting users, not ours, and 18 ratings is still enough to describe a user's taste.

Result: 31,944,676 ratings, 200,948 users, 80,504 movies, 18 genres
(Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western). Note: the genre is called `Children`, not `Children's`.

## Cleaned files
| File | What is inside | Size | When to use it |
|---|---|---|---|
| `sample_ratings.parquet` | userId, movieId, rating for a random 5% of users (seed 42). Each picked user keeps all their ratings. | 10,047 users, 1,599,232 ratings | While building and testing code |
| `clean_ratings.parquet` | userId, movieId, rating for all users | 200,948 users, 31,944,676 ratings | Final runs only |
| `movies_clean.parquet` | movieId + one True/False column per genre | 80,504 movies, 19 columns | Whenever you need genres |

The ratings files have **no genres column**, because a genre list on 32 million rows would use several GB of memory. To get genres, join on `movieId`:
```python
df = ratings.merge(pd.read_parquet(DATA + 'movies_clean.parquet'), on='movieId')
```

## EDA findings
**1. Rating distribution.** Whole-star ratings (3.0, 4.0) are much more common than half stars, and 4.0 is the most common rating. Low ratings are rare. About half of all ratings are 4.0 or higher, so our "liked" rule splits the ratings roughly in half.

**2. Genre popularity.** Drama appears in about 44% of all ratings (14.0 million) and Comedy in about 35% (11.2 million). The least rated genres are Western, Documentary and Film-Noir (under 0.6 million each).

**3. Popular is not the same as liked.** Share of each genre's ratings that are 4.0 or higher:
- Highest: Film-Noir 67%, War 61%, Documentary 57%, Crime 56%, Drama 56%
- Lowest: Horror 41%, Comedy 45%, Children 46%

Niche genres are probably watched mostly by people who already like them, while Horror and Comedy are watched by many casual viewers.

**4. Ratings per user.** Median 73, mean 159, minimum 18, maximum 31,273. A small group of very active users pulls the average up.

## What the EDA means for each part
**Mouaz (clustering):** user profiles use the **share** of each user's ratings in each genre, not counts. With counts, K-Means would separate active users from casual users instead of different tastes. Activity is added as one feature, log(number of ratings), so it counts a little but cannot dominate. Mean rating is also kept, because genres differ in how much they are liked, not only how often they are watched.

**Mohid (association rules):**
- Drama and Comedy will be in a large share of transactions. Rules ending in Drama or Comedy can have high confidence only because these genres are everywhere. **Always check lift**, and compare a rule's confidence with the consequent's own support. These rules are good examples for the "high confidence but lift near 1" part.
- Film-Noir, Western and Documentary are rare, so rules involving them will have low support and may disappear at higher support thresholds. They can still have high lift, so check them at the lower support values.
- Since about half of ratings count as liked, almost every user has liked movies. Users with zero liked movies should be very few.

**Abdallah (PCA and robustness):**
- Drama and Comedy shares will likely vary the most between users, so expect them to have large weights in PC1 or PC2. Check this in the loadings table.
- Rules involving rare genres (Film-Noir, Western, Documentary) are the most likely to be **fragile** in the rule stability check, because their support is close to the threshold. Pay special attention to them.
- Very active users (up to 31,273 ratings) can look like outliers in the PCA plot. If a few points sit far away from everything else, check whether they are these users before treating them as a separate group.
