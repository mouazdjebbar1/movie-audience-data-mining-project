"""Association rule mining. Owner: Mohid."""

import ast
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

RULE_COLUMNS = ["antecedents", "consequents", "support", "confidence", "lift"]


def _genre_list(value):
    """Normalize a genres field and remove categories excluded by our plan."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                parsed = ast.literal_eval(value)
                value = parsed if isinstance(parsed, (list, tuple)) else value
            except (ValueError, SyntaxError):
                pass
        if isinstance(value, str):
            value = value.split("|")
    if value is None or not hasattr(value, "__iter__"):
        return []
    return sorted({str(g).strip() for g in value
                   if pd.notna(g) and str(g).strip() not in
                   ("", "IMAX", "(no genres listed)")})


def build_transactions(ratings_df, user_ids, share_threshold=0.10):
    """Return one True/False genre row per user who liked a movie (rating >= 4).

    For each user, genre share is (liked movies tagged with genre) /
    (all liked movies with at least one valid genre). A multi-genre movie
    counts once in the denominator and once per genre in the numerator.
    """
    if not 0 < share_threshold <= 1:
        raise ValueError("share_threshold must be in (0, 1]")
    user_col = "userId" if "userId" in ratings_df.columns else "user"
    required = {user_col, "rating", "genres"}
    missing = required - set(ratings_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = ratings_df.loc[
        ratings_df[user_col].isin(user_ids) & (ratings_df["rating"] >= 4.0),
        [user_col, "genres"]
    ].copy()
    if df.empty:
        return pd.DataFrame(dtype=bool)

    df["genres"] = df["genres"].map(_genre_list)
    # Discard movies with no usable genres before counting liked movies.
    df = df.loc[df["genres"].map(len) > 0]
    if df.empty:
        return pd.DataFrame(dtype=bool)

    liked_movie_count = df.groupby(user_col).size()
    exploded = df.explode("genres")
    genre_movie_count = exploded.groupby([user_col, "genres"]).size().unstack(fill_value=0)
    shares = genre_movie_count.div(liked_movie_count, axis=0)
    return (shares >= share_threshold).astype(bool).sort_index().sort_index(axis=1)


def mine_rules(ratings_df, user_ids, min_support, min_confidence,share_threshold=0.20):
    """Return association rules as text, ordered by decreasing lift.

    The five columns are always present, including when no rules exist.
    """
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    transactions = build_transactions(ratings_df, user_ids, share_threshold=share_threshold)
    if transactions.empty or transactions.shape[1] < 2:
        return pd.DataFrame(columns=RULE_COLUMNS)

    itemsets = fpgrowth(transactions, min_support=min_support, use_colnames=True)
    if itemsets.empty or not itemsets["itemsets"].map(len).ge(2).any():
        return pd.DataFrame(columns=RULE_COLUMNS)

    rules = association_rules(
        itemsets, num_itemsets=len(transactions),
        metric="confidence", min_threshold=min_confidence
    )
    if rules.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)

    rules = rules[RULE_COLUMNS].copy()
    for col in ("antecedents", "consequents"):
        rules[col] = rules[col].map(lambda genres: ", ".join(sorted(genres)))
    return (rules.sort_values(["lift", "confidence", "support", "antecedents", "consequents"], ascending=[False, False, False, True, True]).reset_index(drop=True))
