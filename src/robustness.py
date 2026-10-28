"""Stability checks for clusters and rules. Owner: Abdallah."""


def cluster_stability(profiles_df, main_labels_df, k):
    """Rerun K-Means with seeds 0-9 and compare to the main labels using ARI."""
    raise NotImplementedError


def rule_stability(ratings_df, user_ids, supports, min_confidence):
    """Run mine_rules at each support value and show which rules stay or disappear."""
    raise NotImplementedError
