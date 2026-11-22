# Discovering Movie Audience Segments and Hidden Viewing Preferences

DSAI 4102 – Applied Data Mining, Fall 2026, UDST

Group: Mouaz Djebbar, Mohid Intikhab, Abdallah Dahmani

## Dataset
MovieLens 32M from GroupLens: https://grouplens.org/datasets/movielens/32m/
Download `ml-32m.zip`, unzip it, and put `ratings.csv` and `movies.csv` in the `data/` folder.
Then run the notebooks in order (see below). They create all the data files in `data/` (cleaned ratings, profiles, cluster labels, rules).
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

`02_clustering` needs the files from `01_data_eda`, and `03`, `04` and `05` need the files from `01` and `02`. Run times: `02` about 3 minutes, `05` about 5 minutes.

Shared functions in `src/`:
| File | Owner | Functions |
|---|---|---|
| `src/data_utils.py` | Mouaz | `liked_ratings_with_genres()`: liked ratings with a `genres` text column, the input `mine_rules()` needs |
| `src/rules.py` | Mohid | `build_transactions()`, `mine_rules()` |
| `src/pca_plot.py` | Abdallah | `plot_pca()` |
| `src/robustness.py` | Abdallah | `cluster_stability()`, `rule_stability()` |

## Shared rules
- Liked = rating >= 4.0
- IMAX and "(no genres listed)" are dropped, leaving 18 genres
- Random seed = 42
- Number of clusters k = 4
- Transactions: a genre is in a user's basket if at least **20%** of the movies they liked have that genre
- Association rules: min support 0.25, min confidence 0.85, keep rules with lift > 1.2

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
| `user_profiles.parquet` | userId + 20 **scaled** features (made in `02_clustering`) | 200,948 users | K-Means, PCA, cluster stability |
| `user_profiles_raw.parquet` | Same 20 features, **not scaled** | 200,948 users | Describing the clusters in easy numbers |
| `cluster_labels.csv` | userId, cluster (0 to 3) (made in `02_clustering`) | 200,948 users | Anything per cluster |
| `cluster_rules.csv` | All rules with lift > 1.2, with a `group` column (made in `05_cluster_rules`) | 61 rules | Comparing rules, rule stability |

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

## User profiles (02_clustering)
Each user is described by 20 features:
- **18 genre shares**: the share of the user's ratings that include each genre (what they watch). Shares, not counts, so active and casual users are compared fairly.
- **mean_rating**: the user's average rating (how much they enjoy what they watch).
- **log_n_ratings**: log of the user's number of ratings (how active they are). Log, so very active users cannot dominate.

All 20 features are scaled with `StandardScaler` (mean 0, standard deviation 1), because K-Means uses distances and features with bigger numbers would otherwise control them.

## Clustering results (02_clustering)
We ran K-Means (`n_init=10`, seed 42) for k = 2 to 10 and chose **k = 4**:

| k | Silhouette (higher is better) | Davies-Bouldin (lower is better) | Smallest cluster |
|---|---|---|---|
| 2 | 0.123 | 2.437 | 94,759 |
| 3 | 0.125 | 2.105 | 35,331 |
| **4** | **0.121** | **2.033** | **20,611** |
| 5 | 0.096 | 2.139 | 19,302 |
| 7 | 0.093 | 1.943 | 4,498 |
| 10 | 0.099 | 1.826 | 3,079 |

- **Elbow:** inertia drops by 10% (k=2 to 3) and 7% (k=3 to 4), then only 3-5% per extra cluster.
- **Silhouette:** k=2 to 4 are almost equal, then it falls sharply at k=5.
- **Davies-Bouldin:** k=4 is lower than k=2, 3 and 5.
- **Cluster size:** from k=7 the smallest cluster is about 2% of users or less, too small to be a useful audience group.
- **Limitation:** a silhouette of about 0.12 means the clusters overlap. User tastes change gradually rather than falling into fully separate groups.

**The 4 audience groups** (genre differences are percentage points compared with the average user):

| Cluster | Name | Users | Watches more | Watches less | Other |
|---|---|---|---|---|---|
| 0 | Drama & crime fans | 45,933 (23%) | Drama +12, Crime +9, Thriller +7, Mystery +4 | Comedy, Adventure | Highest mean rating (3.87) |
| 1 | Comedy & romance fans | 71,577 (36%) | Comedy +9, Romance +7, Drama +4 | Action -10, Adventure, Sci-Fi | Lowest mean rating (3.57), most active (median 85 ratings) |
| 2 | Family & animation fans | 20,611 (10%) | Children +17, Animation +15, Adventure +14, Fantasy +12, Musical +6 | Drama -11, Thriller -11 | |
| 3 | Action & sci-fi fans | 62,827 (31%) | Action +15, Sci-Fi +10, Adventure +9, Thriller +6 | Drama, Comedy, Romance | |

No group forms around rare genres such as Documentary or Film-Noir (under 2% share in every cluster).
Charts: `figures/kmeans_choose_k.png`, `figures/cluster_genre_differences.png`.

## Association rules: choosing the share threshold
`mine_rules()` originally used a 10% share threshold. Tested on 10,000 users (support 0.25, confidence 0.85):

| Share threshold | Genres per user | Rules | Best lift |
|---|---|---|---|
| 10% | 8.8 | 3,267 | 1.27 |
| **20%** | **5.4** | **59** | **1.52** |
| 30% | 3.2 | 1 | 1.00 |

At 10%, almost every user has almost every genre (Drama 99%, Comedy 95%, Thriller 91%), so the rules are long, repetitive and weak. At 30%, only Drama is common enough to pass 25% support. We use **20%**, now the default `share_threshold` in `mine_rules()`.

## Cluster-specific rules (05_cluster_rules)
Rules with lift > 1.2 per group:

| Group | Users | Rules found | Rules with lift > 1.2 |
|---|---|---|---|
| All users | 200,948 | 61 | 30 |
| Drama & crime | 45,933 | 71 | 6 |
| Comedy & romance | 71,577 | 42 | 0 |
| Family & animation | 20,611 | 232 | 25 |
| Action & sci-fi | 62,827 | 327 | 0 |

To compare fairly, every rule found in any group was measured in every group (support / lift):

| Rule | All users | Drama & crime | Comedy & romance | Family & animation | Action & sci-fi |
|---|---|---|---|---|---|
| Action, Sci-Fi -> Adventure | 29% / 1.51 | 11% / 1.75 | 6% / 2.11 | 37% / 1.06 | **65% / 1.05** |
| Crime -> Thriller | 41% / 1.26 | **77% / 1.04** | 22% / 1.44 | 8% / 1.93 | 48% / 1.05 |
| Animation, Comedy -> Children | 6% / 8.11 | 0% | 1% | **43% / 1.39** | 2% |
| Romance -> Comedy, Drama | 33% / 1.20 | 17% / 1.26 | **62% / 1.02** | 40% / 1.11 | 11% / 1.29 |

**Findings**
1. **In the cluster that matches a rule's genres, the rule is very common but its lift is close to 1.** Almost everyone in that group already likes all those genres, so knowing one tells you nothing extra.
2. **In the other clusters, the same rule is rare but has high lift.** It describes a small subgroup, e.g. 6% of Comedy & romance fans with a hidden action side (lift 2.11).
3. **For all users together, the lift is a mix of both.** Most rules found on all users mainly describe the audience groups, not a general link between genres.
4. **Animation, Comedy -> Children only appears inside the Family cluster** (43% support there, 6% overall). With a 25% minimum support it cannot be found without clustering.
5. **"0 rules" in a cluster does not mean no patterns.** Comedy & romance and Action & sci-fi have strong rules, but each covers less than 25% of the group, or is already true for almost everyone in it.
6. **Lift is unreliable when support is tiny** (e.g. lift 80.9 with 0% support in Drama & crime), so lift is always read together with support.
7. 222 users never rated a movie 4.0 or higher, so they have no transaction (200,726 of 200,948).

Chart: `figures/rules_across_clusters.png`.
