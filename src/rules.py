"""Association rule mining. Owner: Mohid."""


def build_transactions(ratings_df, user_ids, share_threshold=0.10):
    """Return one row per user and one True/False column per genre.
    A genre is True if at least share_threshold of the user's liked movies have it."""
    raise NotImplementedError


def mine_rules(ratings_df, user_ids, min_support, min_confidence):
    """Return a DataFrame with columns:
    antecedents, consequents, support, confidence, lift.
    Genres are text, sorted and joined with ', '. Empty table (same columns) if no rules."""
    raise NotImplementedError
