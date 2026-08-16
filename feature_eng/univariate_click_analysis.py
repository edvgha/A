"""
univariate_click_analysis.py
============================
Quick univariate screen of features against a binary `click` column:
for every feature, compare its distribution in click=1 vs click=0 rows.

Numeric features get:
    group means, mean difference, Cohen's d, point-biserial r,
    Welch t-test p, Mann-Whitney U p, single-feature AUC
Categorical features get:
    chi-square p, Cramer's V, CTR of best / worst level

Both tables include Benjamini-Hochberg adjusted p-values (you are testing
many features at once) and are sorted by effect size, not p-value.

Typical use
-----------
    from univariate_click_analysis import (
        univariate_click_analysis, plot_top_numeric_features)

    num_res, cat_res = univariate_click_analysis(df, click_col="click")
    print(num_res.head(15))
    print(cat_res)
    plot_top_numeric_features(df, num_res, click_col="click", top_n=6)

Notes
-----
* Non-numeric columns (string / object / category) and bool are treated
  as categorical.
  Integer-coded categoricals (0/1 flags, ids, bucketed levels) should be
  passed explicitly via `categorical_cols=[...]`.
* With large n almost everything is "significant" -- rank by |cohens_d|,
  Cramer's V or |auc - 0.5|, and use p-values only as a sanity filter.
* This is univariate: it ignores interactions and feature correlations.

Run this file directly (`python univariate_click_analysis.py`) to see a
demo on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------- helpers
def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def _cohens_d(x1: np.ndarray, x0: np.ndarray) -> float:
    """Standardized mean difference (pooled SD)."""
    n1, n0 = len(x1), len(x0)
    s1, s0 = x1.std(ddof=1), x0.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    return 0.0 if pooled == 0 else (x1.mean() - x0.mean()) / pooled


def _cramers_v(chi2: float, n: int, shape: tuple) -> float:
    """Cramer's V effect size for a contingency table (0..1)."""
    r, k = shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


# ------------------------------------------------------------- main API
def univariate_click_analysis(
    df: pd.DataFrame,
    click_col: str = "click",
    feature_cols: list | None = None,
    categorical_cols: list | None = None,
    max_levels: int = 20,
):
    """
    Compare every feature's distribution between click=1 and click=0 rows.

    Parameters
    ----------
    df : DataFrame containing the features and the click column.
    click_col : name of the binary 0/1 target column.
    feature_cols : which columns to analyze (default: all except click_col).
    categorical_cols : columns to force-treat as categorical
        (default: object / category / bool dtypes).
    max_levels : rare categories beyond this count are bucketed as __OTHER__.

    Returns
    -------
    (numeric_results, categorical_results) : two DataFrames sorted by
    effect size (|Cohen's d| and Cramer's V respectively).
    """
    df = df.copy()
    df[click_col] = pd.to_numeric(df[click_col]).astype(int)
    if not set(df[click_col].unique()) <= {0, 1}:
        raise ValueError(f"'{click_col}' must contain only 0/1 values")

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != click_col]
    if categorical_cols is None:
        # anything non-numeric (string, object, category) plus bool
        categorical_cols = [
            c for c in feature_cols
            if not pd.api.types.is_numeric_dtype(df[c])
            or pd.api.types.is_bool_dtype(df[c])
        ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # ---------------------------------------------- numeric features
    num_rows = []
    for f in numeric_cols:
        sub = df[[f, click_col]].dropna()
        x1 = sub.loc[sub[click_col] == 1, f].astype(float).to_numpy()
        x0 = sub.loc[sub[click_col] == 0, f].astype(float).to_numpy()
        if len(x1) < 2 or len(x0) < 2 or sub[f].nunique() < 2:
            continue  # constant feature or missing class -> nothing to test

        _, p_t = stats.ttest_ind(x1, x0, equal_var=False)          # Welch
        u_stat, p_u = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        r_pb, _ = stats.pointbiserialr(
            sub[click_col].to_numpy(), sub[f].astype(float).to_numpy()
        )
        # U / (n1*n0) = P(value | click=1  >  value | click=0), i.e. the AUC
        # this single feature would achieve as a ranking score.
        auc = u_stat / (len(x1) * len(x0))
        diff = x1.mean() - x0.mean()

        num_rows.append({
            "feature": f,
            "n_used": len(sub),
            "mean_click1": x1.mean(),
            "mean_click0": x0.mean(),
            "mean_diff": diff,
            "direction": "higher when clicked" if diff > 0 else "lower when clicked",
            "cohens_d": _cohens_d(x1, x0),
            "auc_single_feature": auc,
            "pointbiserial_r": r_pb,
            "p_ttest_welch": p_t,
            "p_mannwhitney": p_u,
        })

    num_res = pd.DataFrame(num_rows)
    if not num_res.empty:
        num_res["p_adj_BH"] = _benjamini_hochberg(num_res["p_mannwhitney"])
        num_res = num_res.sort_values(
            "cohens_d", key=lambda s: s.abs(), ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------ categorical features
    cat_rows = []
    for f in categorical_cols:
        sub = df[[f, click_col]].dropna()
        if sub[f].nunique() < 2 or sub[click_col].nunique() < 2:
            continue

        lvl = sub[f].astype(str)
        if lvl.nunique() > max_levels:                 # bucket rare levels
            top = lvl.value_counts().nlargest(max_levels - 1).index
            lvl = lvl.where(lvl.isin(top), "__OTHER__")

        ct = pd.crosstab(lvl, sub[click_col])
        chi2, p_chi, _, expected = stats.chi2_contingency(ct)

        ctr = sub.assign(_lvl=lvl).groupby("_lvl")[click_col].agg(["mean", "count"])
        best, worst = ctr["mean"].idxmax(), ctr["mean"].idxmin()

        cat_rows.append({
            "feature": f,
            "n_used": len(sub),
            "n_levels": ct.shape[0],
            "cramers_v": _cramers_v(chi2, len(sub), ct.shape),
            "p_chi2": p_chi,
            "best_level": f"{best} (CTR={ctr.loc[best, 'mean']:.3f}, n={int(ctr.loc[best, 'count'])})",
            "worst_level": f"{worst} (CTR={ctr.loc[worst, 'mean']:.3f}, n={int(ctr.loc[worst, 'count'])})",
            # chi-square is unreliable if many expected cell counts are < 5
            "pct_expected_cells_lt5": (expected < 5).mean(),
        })

    cat_res = pd.DataFrame(cat_rows)
    if not cat_res.empty:
        cat_res["p_adj_BH"] = _benjamini_hochberg(cat_res["p_chi2"])
        cat_res = cat_res.sort_values("cramers_v", ascending=False).reset_index(drop=True)

    return num_res, cat_res


def plot_top_numeric_features(
    df: pd.DataFrame,
    num_res: pd.DataFrame,
    click_col: str = "click",
    top_n: int = 6,
    bins: int = 30,
    save_path: str | None = None,
):
    """Overlaid density histograms (click=0 vs click=1) for the strongest features."""
    import matplotlib.pyplot as plt

    feats = num_res["feature"].head(top_n).tolist()
    if not feats:
        return None
    ncols = min(3, len(feats))
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for ax, f in zip(axes.flat, feats):
        sub = df[[f, click_col]].dropna()
        for val, color, label in [(0, "tab:gray", "click=0"), (1, "tab:red", "click=1")]:
            ax.hist(sub.loc[sub[click_col] == val, f].astype(float),
                    bins=bins, density=True, alpha=0.5, color=color, label=label)
        d = num_res.loc[num_res["feature"] == f, "cohens_d"].iloc[0]
        ax.set_title(f"{f}   (Cohen's d = {d:+.2f})")
        ax.legend()

    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 20_000

    price = rng.gamma(4, 25, n)                                # hurts clicks
    rating = np.clip(rng.normal(4.0, 0.6, n), 1, 5)            # helps clicks
    delivery_days = rng.integers(1, 10, n).astype(float)       # mildly hurts
    noise = rng.normal(0, 1, n)                                # irrelevant
    brand = rng.choice(["acme", "globex", "initech", "umbrella"], n, p=[.4, .3, .2, .1])
    device = rng.choice(["mobile", "desktop", "tablet"], n, p=[.6, .3, .1])  # irrelevant

    brand_eff = {"acme": 0.3, "globex": 0.0, "initech": -0.3, "umbrella": 0.6}
    logit = (-2.2
             - 0.012 * (price - 100)
             + 0.9 * (rating - 4.0)
             - 0.10 * (delivery_days - 5)
             + pd.Series(brand).map(brand_eff).to_numpy())
    click = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    demo = pd.DataFrame({
        "f_price": price, "f_rating": rating, "f_delivery_days": delivery_days,
        "f_noise": noise, "brand": brand, "device": device, "click": click,
    })
    print(f"demo data: {n} rows, overall CTR = {demo['click'].mean():.3f}\n")

    num_res, cat_res = univariate_click_analysis(demo, click_col="click")

    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.4f}".format):
        print("=== numeric features (sorted by |Cohen's d|) ===")
        print(num_res, "\n")
        print("=== categorical features (sorted by Cramer's V) ===")
        print(cat_res)

    plot_top_numeric_features(demo, num_res, save_path="univariate_top_features.png")
    print("\nsaved plot -> univariate_top_features.png")
