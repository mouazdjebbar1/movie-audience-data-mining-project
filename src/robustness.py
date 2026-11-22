"""Stability checks for clusters and rules. Owner: Abdallah."""

from importlib import import_module
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def _detect_column(df: pd.DataFrame, candidates: Iterable[str], role: str) -> str:
    """Find the first matching column name in a DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find {role} column among {list(candidates)}. Available: {list(df.columns)}"
    )


def _resolve_mine_rules(mine_rules_fn: Optional[Callable] = None) -> Callable:
    """Locate Mohid's mine_rules function across possible module paths."""
    if mine_rules_fn is not None:
        return mine_rules_fn
    for module_path in ("src.rules", "rules"):
        try:
            mod = import_module(module_path)
            if hasattr(mod, "mine_rules"):
                return getattr(mod, "mine_rules")
        except ImportError:
            continue
    raise ImportError(
        "Could not import mine_rules from src.rules or rules. Pass mine_rules_fn explicitly."
    )


def _format_itemset(items) -> str:
    """Format an itemset or iterable cleanly for rule strings."""
    if isinstance(items, (set, frozenset, list, tuple)):
        ordered = sorted(str(x) for x in items)
        return ", ".join(ordered)
    return str(items)


def _format_rule_label(row: pd.Series) -> str:
    """Format an association rule as 'antecedents → consequents'."""
    ant_col = "antecedents" if "antecedents" in row.index else "antecedent"
    con_col = "consequents" if "consequents" in row.index else "consequent"
    return f"{_format_itemset(row[ant_col])} → {_format_itemset(row[con_col])}"


def _format_support_key(s: float) -> str:
    """Format a support threshold cleanly for column names (e.g., 0.01 -> '0.01')."""
    return f"{s:.4f}".rstrip("0").rstrip(".")


def cluster_stability(
    profiles_df: pd.DataFrame,
    main_labels_df: pd.DataFrame,
    k: int,
    seeds: Iterable[int] = range(10),
    max_users: int = 50_000,
    random_state: int = 42,
    n_init: int = 10,
    return_tables: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Rerun K-Means with seeds 0-9 and compare to the main labels using ARI."""
    if k < 2:
        raise ValueError("k must be at least 2 for clustering stability checks.")

    prof_user_col = _detect_column(profiles_df, ("userId", "user_id", "user"), "user ID")
    lbl_user_col = _detect_column(main_labels_df, ("userId", "user_id", "user"), "user ID")
    cluster_col = _detect_column(main_labels_df, ("cluster", "cluster_label", "label"), "cluster label")

    labels_clean = (
        main_labels_df[[lbl_user_col, cluster_col]]
        .drop_duplicates(subset=[lbl_user_col])
        .copy()
    )
    if lbl_user_col != prof_user_col:
        labels_clean = labels_clean.rename(columns={lbl_user_col: prof_user_col})

    merged = profiles_df.merge(labels_clean, on=prof_user_col, how="inner")
    if merged.empty:
        raise ValueError("No overlapping users between profiles_df and main_labels_df.")

    if len(merged) > max_users:
        print(
            f"Sampling {max_users:,} of {len(merged):,} users with seed={random_state} for runtime stability."
        )
        merged = merged.sample(n=max_users, random_state=random_state).reset_index(drop=True)

    non_feature_cols = {prof_user_col, cluster_col}
    feature_cols = [
        c
        for c in merged.columns
        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(merged[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in profiles_df.")

    X = merged[feature_cols].to_numpy(dtype=float)
    main_labels = merged[cluster_col].to_numpy()

    ari_rows: List[Dict[str, float | int]] = []
    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=int(seed), n_init=n_init)
        run_labels = km.fit_predict(X)
        ari_val = float(adjusted_rand_score(main_labels, run_labels))
        ari_rows.append({"seed": int(seed), "ari": ari_val})

    ari_df = pd.DataFrame(ari_rows)
    mean_ari = float(ari_df["ari"].mean())
    min_ari = float(ari_df["ari"].min())
    print(f"Cluster stability across seeds (k={k}) -> Mean ARI: {mean_ari:.4f} | Min ARI: {min_ari:.4f}")

    comparison_tables: Dict[str, pd.DataFrame] = {}
    if k > 2:
        km_minus = KMeans(n_clusters=k - 1, random_state=random_state, n_init=n_init)
        labels_minus = km_minus.fit_predict(X)
        ct_minus = pd.crosstab(
            pd.Series(main_labels, name=f"main_k_{k}"),
            pd.Series(labels_minus, name=f"k_{k - 1}"),
        )
        comparison_tables[f"k_minus_1 ({k - 1})"] = ct_minus
        print(f"\n--- Main clusters (k={k}) vs. k-1 ({k - 1}) ---")
        print(ct_minus)

    km_plus = KMeans(n_clusters=k + 1, random_state=random_state, n_init=n_init)
    labels_plus = km_plus.fit_predict(X)
    ct_plus = pd.crosstab(
        pd.Series(main_labels, name=f"main_k_{k}"),
        pd.Series(labels_plus, name=f"k_{k + 1}"),
    )
    comparison_tables[f"k_plus_1 ({k + 1})"] = ct_plus
    print(f"\n--- Main clusters (k={k}) vs. k+1 ({k + 1}) ---")
    print(ct_plus)

    if return_tables:
        return ari_df, comparison_tables
    return ari_df


def rule_stability(
    ratings_df: pd.DataFrame,
    user_ids: Iterable,
    supports: Iterable[float],
    min_confidence: float,
    mine_rules_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Run mine_rules at each support value and show which rules stay or disappear."""
    support_list = [float(s) for s in supports]
    if not support_list:
        raise ValueError("supports must contain at least one support value.")

    rules_func = _resolve_mine_rules(mine_rules_fn)
    user_id_list = list(user_ids)

    per_support_maps: Dict[float, Dict[str, float]] = {}
    all_rules_ordered: List[str] = []
    seen_rules = set()

    for s in support_list:
        res_df = rules_func(ratings_df, user_id_list, min_support=s, min_confidence=min_confidence)
        rule_to_lift: Dict[str, float] = {}
        if res_df is not None and not res_df.empty:
            for _, row in res_df.iterrows():
                label = _format_rule_label(row)
                lift_val = (
                    float(row["lift"])
                    if "lift" in row.index and pd.notna(row["lift"])
                    else np.nan
                )
                rule_to_lift[label] = lift_val
                if label not in seen_rules:
                    seen_rules.add(label)
                    all_rules_ordered.append(label)
        per_support_maps[s] = rule_to_lift

    rows: List[Dict] = []
    stable_count = 0
    fragile_count = 0

    sorted_supports = sorted(support_list)
    lowest_s = sorted_supports[0]

    for rule_label in all_rules_ordered:
        row_dict: Dict = {"rule": rule_label}
        present_flags: List[bool] = []
        for s in support_list:
            key = _format_support_key(s)
            is_present = rule_label in per_support_maps[s]
            row_dict[f"present_{key}"] = is_present
            row_dict[f"lift_{key}"] = per_support_maps[s].get(rule_label, np.nan)
            present_flags.append(is_present)

        is_stable = all(present_flags)
        is_fragile = (rule_label in per_support_maps[lowest_s]) and (not is_stable)
        row_dict["stable"] = is_stable
        row_dict["fragile"] = is_fragile

        if is_stable:
            stable_count += 1
        if is_fragile:
            fragile_count += 1

        rows.append(row_dict)

    print(
        f"Rule stability across supports {support_list} -> Stable rules: {stable_count} | Fragile rules: {fragile_count}"
    )

    cols = ["rule"]
    for s in support_list:
        key = _format_support_key(s)
        cols.extend([f"present_{key}", f"lift_{key}"])
    cols.extend(["stable", "fragile"])

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]
