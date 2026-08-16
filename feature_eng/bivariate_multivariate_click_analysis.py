"""
bivariate_multivariate_click_analysis.py
========================================
Stage 2 of the click analysis. Stage 1 (univariate) looked at each feature
alone -> marginal association with click. This stage looks at:

  BIVARIATE  : feature <-> feature structure (correlation, redundancy)
  MULTIVARIATE: click ~ ALL features jointly (adjusted effects, nonlinearity,
                interactions, importance)

Pipeline
--------
1. Mixed-type association matrix   Pearson (num-num), correlation ratio eta
                                   (num-cat), Cramer's V (cat-cat)
2. VIF                             variance inflation -> collinearity severity
3. Hierarchical feature clustering group "tug-of-war" features (dist = 1-assoc)
4. Logistic regression (statsmodels) adjusted linear effects + inference
5. LightGBM + SHAP                 nonlinearity, interactions, importance
6. Permutation importance          per feature AND per cluster (joint shuffle)
7. Drop-cluster retrain            unique (non-redundant) signal of a cluster
8. Likelihood-ratio test           formal test for a chosen interaction pair
9. Ridge vs Lasso mini-demo        how regularization behaves under collinearity

Run directly (`python bivariate_multivariate_click_analysis.py`) for a demo
whose data generating process deliberately contains:
  * f_price_dup   ~ 0.97-correlated copy of f_price, NO effect of its own
                    -> credit tug-of-war
  * f_popularity  ~ 0.75-correlated with f_rating,   NO effect of its own
                    -> proxy / confounded feature
  * f_novelty     inverted-U effect (best in the middle)
                    -> invisible to univariate mean/rank tests AND to linear logit
  * delivery x device interaction (delivery only matters on mobile)
                    -> invisible to any single-feature view
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ======================================================================
# 1. BIVARIATE: mixed-type association matrix
# ======================================================================
def correlation_ratio(cat, y) -> float:
    """eta = sqrt(between-group SS / total SS): association categorical->numeric.
    For a binary y this equals |point-biserial r|; it is the ANOVA R."""
    y = np.asarray(y, float)
    cat = np.asarray(cat)
    ybar = y.mean()
    sst = ((y - ybar) ** 2).sum()
    if sst == 0:
        return 0.0
    ssb = 0.0
    for g in np.unique(cat):
        yg = y[cat == g]
        ssb += len(yg) * (yg.mean() - ybar) ** 2
    return float(np.sqrt(ssb / sst))


def cramers_v_pair(a, b) -> float:
    """Cramer's V between two categorical series (no Yates correction:
    we want an unbiased *measure*, not a conservative test)."""
    ct = pd.crosstab(pd.Series(a).astype(str), pd.Series(b).astype(str))
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    denom = n * (min(ct.shape) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def association_matrix(df: pd.DataFrame, numeric_cols, categorical_cols) -> pd.DataFrame:
    """Symmetric matrix in [0,1]. num-num: |Pearson|; num-cat: eta; cat-cat: V.
    All three reduce to the same 'shared variance' family, so mixing them on
    one scale is meaningful for *screening* redundancy (not exact equivalence)."""
    cols = list(numeric_cols) + list(categorical_cols)
    A = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, ci in enumerate(cols):
        for cj in cols[i + 1:]:
            ci_num, cj_num = ci in numeric_cols, cj in numeric_cols
            sub = df[[ci, cj]].dropna()
            if ci_num and cj_num:
                v = abs(stats.pearsonr(sub[ci], sub[cj])[0])
            elif ci_num != cj_num:                       # mixed pair
                num, cat = (ci, cj) if ci_num else (cj, ci)
                v = correlation_ratio(sub[cat], sub[num])
            else:
                v = cramers_v_pair(sub[ci], sub[cj])
            A.loc[ci, cj] = A.loc[cj, ci] = v
    return A


# ======================================================================
# 2. VIF (computed on the encoded design matrix)
# ======================================================================
def compute_vif(X: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 - R2_j), R2_j from regressing column j on all others.
    sqrt(VIF_j) = factor by which collinearity inflates SE(beta_j)."""
    out = {}
    for j, col in enumerate(X.columns):
        others = X.drop(columns=col)
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        out[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(out, name="VIF").sort_values(ascending=False)


# ======================================================================
# 3. Feature clustering on distance = 1 - association
# ======================================================================
def cluster_features(assoc: pd.DataFrame, threshold: float = 0.7):
    """Average-linkage hierarchical clustering; features with association
    > threshold end up in one cluster. Returns (labels Series, linkage Z)."""
    D = 1.0 - assoc.values
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    Z = hierarchy.linkage(squareform(D, checks=False), method="average")
    labels = hierarchy.fcluster(Z, t=1.0 - threshold, criterion="distance")
    return pd.Series(labels, index=assoc.index, name="cluster"), Z


# ======================================================================
# helpers: encoding (shared by logistic, GBM, importances)
# ======================================================================
def encode_features(df, numeric_cols, categorical_cols, standardize=True):
    """z-score numerics (comparable coefficients), one-hot categoricals
    (drop_first -> reference category, avoids the exact-collinearity trap).
    Returns (X, mapping original feature -> its encoded column list)."""
    parts, mapping = [], {}
    for c in numeric_cols:
        x = df[c].astype(float)
        parts.append(((x - x.mean()) / x.std(ddof=0) if standardize else x).rename(c))
        mapping[c] = [c]
    for c in categorical_cols:
        d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True).astype(float)
        parts.append(d)
        mapping[c] = list(d.columns)
    X = pd.concat(parts, axis=1)
    return X, mapping


# ======================================================================
# 4. Multivariate additive: logistic regression with inference
# ======================================================================
def fit_logistic_inference(X: pd.DataFrame, y: pd.Series):
    """statsmodels Logit -> tidy table with coef, SE, z, p, odds ratio, CI.
    Numerics are z-scored, so exp(coef) = odds multiplier per +1 SD,
    holding every other column fixed (that clause IS the adjustment)."""
    import statsmodels.api as sm
    res = sm.Logit(y.astype(float), sm.add_constant(X)).fit(disp=0)
    tab = pd.DataFrame({
        "coef": res.params, "SE": res.bse, "z": res.tvalues, "p": res.pvalues,
        "odds_ratio": np.exp(res.params),
        "OR_lo95": np.exp(res.params - 1.96 * res.bse),
        "OR_hi95": np.exp(res.params + 1.96 * res.bse),
    })
    return res, tab.drop(index="const")


# ======================================================================
# 5. Flexible model: LightGBM + SHAP
# ======================================================================
def fit_gbm(X_tr, y_tr, X_te, y_te, seed=0):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=seed, verbose=-1)
    model.fit(X_tr, y_tr)
    return model, roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])


def shap_values_of(model, X):
    """Return (n, p) SHAP matrix in log-odds units, robust to SHAP versions
    that return a list [class0, class1] instead of one array."""
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:                      # (n, p, 2) layout in some versions
        sv = sv[:, :, 1]
    return sv


def shap_interaction_ranking(model, X, top=6):
    """Mean |SHAP interaction value| per pair (i<j) -> top interacting pairs.
    Exact for trees; O(n * p^2), so pass a subsample."""
    import shap
    iv = shap.TreeExplainer(model).shap_interaction_values(X)
    if isinstance(iv, list):
        iv = iv[1]
    M = np.abs(iv).mean(axis=0)           # (p, p)
    rows = []
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append((cols[i], cols[j], M[i, j]))
    return (pd.DataFrame(rows, columns=["feat_a", "feat_b", "mean_abs_interaction"])
            .sort_values("mean_abs_interaction", ascending=False).head(top)
            .reset_index(drop=True))


# ======================================================================
# 6. Permutation importance: single columns and whole clusters
# ======================================================================
def permutation_importance_auc(model, X, y, col_groups: dict, n_repeats=5, seed=0):
    """AUC drop when the columns of a group are shuffled JOINTLY with one
    shared row permutation. Joint shuffle preserves the group's internal
    correlation while severing group<->target and group<->rest links --
    this is the fix for correlated features hiding behind each other."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for name, cols in col_groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp.loc[:, cols] = X[cols].to_numpy()[perm]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[name] = float(np.mean(drops))
    return base, pd.Series(out, name="auc_drop").sort_values(ascending=False)


# ======================================================================
# 7. Drop-cluster retrain: unique information of a feature block
# ======================================================================
def drop_cluster_importance(X_tr, y_tr, X_te, y_te, col_groups: dict, seed=0):
    """Retrain WITHOUT the block -> AUC loss = signal no other feature can
    replace. Near zero for a block with a correlated twin outside it
    (redundant), large for irreplaceable blocks."""
    _, base = fit_gbm(X_tr, y_tr, X_te, y_te, seed)
    out = {}
    for name, cols in col_groups.items():
        keep = [c for c in X_tr.columns if c not in cols]
        _, auc = fit_gbm(X_tr[keep], y_tr, X_te[keep], y_te, seed)
        out[name] = base - auc
    return base, pd.Series(out, name="auc_loss_when_dropped").sort_values(ascending=False)


# ======================================================================
# 8. Likelihood-ratio test for one interaction
# ======================================================================
def interaction_lrt(df, click_col, f_a, f_b, categorical_cols):
    """H0: click ~ A + B (additive)  vs  H1: click ~ A * B (with interaction).
    LRT = 2*(ll_full - ll_reduced) ~ chi2 with df = #interaction terms."""
    import statsmodels.formula.api as smf
    def term(f):
        return f"C({f})" if f in categorical_cols else f
    a, b = term(f_a), term(f_b)
    red = smf.logit(f"{click_col} ~ {a} + {b}", data=df).fit(disp=0)
    full = smf.logit(f"{click_col} ~ {a} * {b}", data=df).fit(disp=0)
    lam = 2 * (full.llf - red.llf)
    dof = int(full.df_model - red.df_model)
    return {"feat_a": f_a, "feat_b": f_b, "LRT": lam, "dof": dof,
            "p_interaction": float(stats.chi2.sf(lam, dof))}


# ======================================================================
# 9. Ridge vs Lasso under collinearity (mini-demo)
# ======================================================================
def regularization_tug_of_war_demo(X2: pd.DataFrame, y):
    """Fit logistic on ONLY a collinear pair. Ridge (L2) splits the credit
    ~equally and stably; Lasso (L1) arbitrarily picks one and zeroes the other."""
    ridge = LogisticRegression(C=1.0, l1_ratio=0.0, max_iter=2000).fit(X2, y)
    lasso = LogisticRegression(C=0.05, l1_ratio=1.0, solver="liblinear",
                               max_iter=2000).fit(X2, y)
    return pd.DataFrame({"ridge_coef": ridge.coef_[0], "lasso_coef": lasso.coef_[0]},
                        index=X2.columns)


# ======================================================================
# quick univariate echo (stage-1 numbers for the comparison table)
# ======================================================================
def quick_univariate(df, numeric_cols, click_col):
    rows = {}
    y = df[click_col].to_numpy()
    for f in numeric_cols:
        x = df[f].astype(float).to_numpy()
        x1, x0 = x[y == 1], x[y == 0]
        sp = np.sqrt(((len(x1) - 1) * x1.std(ddof=1) ** 2 +
                      (len(x0) - 1) * x0.std(ddof=1) ** 2) / (len(x) - 2))
        u, _ = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        rows[f] = {"cohens_d": (x1.mean() - x0.mean()) / sp,
                   "auc_uni": u / (len(x1) * len(x0))}
    return pd.DataFrame(rows).T


# ======================================================================
# DEMO
# ======================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    rng = np.random.default_rng(7)
    n = 20_000

    # ----- features ----------------------------------------------------
    price = rng.gamma(4, 25, n)                                   # real effect
    price_dup = price * 7.2 + rng.normal(0, 90, n)                # twin, NO own effect
    rating = np.clip(rng.normal(4.0, 0.6, n), 1, 5)               # real effect
    z_r = (rating - rating.mean()) / rating.std()
    popularity = 50 + 15 * (0.75 * z_r + rng.normal(0, 0.6614, n))  # proxy, NO own effect
    novelty = rng.uniform(0, 10, n)                               # inverted-U effect
    delivery = rng.integers(1, 10, n).astype(float)               # effect ONLY on mobile
    noise = rng.normal(0, 1, n)                                   # nothing
    brand = rng.choice(["acme", "globex", "initech", "umbrella"], n, p=[.4, .3, .2, .1])
    device = rng.choice(["mobile", "desktop", "tablet"], n, p=[.6, .3, .1])

    # ----- true click mechanism (what the analysis should recover) -----
    brand_eff = {"acme": 0.3, "globex": 0.0, "initech": -0.3, "umbrella": 0.6}
    logit = (-2.1
             - 0.012 * (price - 100)                      # linear price effect
             + 0.9 * (rating - 4.0)                       # linear rating effect
             + 0.5 - 0.055 * (novelty - 5.0) ** 2         # inverted U in novelty
             - 0.16 * (delivery - 5.0) * (device == "mobile")   # pure interaction
             + pd.Series(brand).map(brand_eff).to_numpy())
    click = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame({"f_price": price, "f_price_dup": price_dup,
                       "f_rating": rating, "f_popularity": popularity,
                       "f_novelty": novelty, "f_delivery_days": delivery,
                       "f_noise": noise, "brand": brand, "device": device,
                       "click": click})
    numeric_cols = ["f_price", "f_price_dup", "f_rating", "f_popularity",
                    "f_novelty", "f_delivery_days", "f_noise"]
    categorical_cols = ["brand", "device"]
    print(f"n={n}, CTR={df['click'].mean():.3f}")

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
    fmt = "{:.4f}".format

    # ----- 1. association matrix + heatmap -----------------------------
    A = association_matrix(df, numeric_cols, categorical_cols)
    print("\n=== [1] association matrix (|Pearson| / eta / Cramer's V) ===")
    print(A.round(2))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(A.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(A))); ax.set_xticklabels(A.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(A))); ax.set_yticklabels(A.index)
    for i in range(len(A)):
        for j in range(len(A)):
            ax.text(j, i, f"{A.values[i, j]:.2f}", ha="center", va="center",
                    color="w" if A.values[i, j] < 0.6 else "k", fontsize=7)
    fig.colorbar(im); fig.tight_layout(); fig.savefig("assoc_heatmap.png", dpi=120); plt.close(fig)

    # ----- 3. clustering (before VIF so we can report cluster ids) -----
    clusters, Z = cluster_features(A, threshold=0.7)
    print("\n=== [3] feature clusters (association > 0.7 merged) ===")
    for cid in sorted(clusters.unique()):
        print(f"  cluster {cid}: {list(clusters.index[clusters == cid])}")
    fig, ax = plt.subplots(figsize=(8, 4))
    hierarchy.dendrogram(Z, labels=list(A.index), ax=ax, color_threshold=0.3)
    ax.axhline(0.3, ls="--", c="gray"); ax.set_ylabel("distance = 1 - association")
    fig.tight_layout(); fig.savefig("feature_dendrogram.png", dpi=120); plt.close(fig)

    # ----- encoding shared by every model ------------------------------
    X, mapping = encode_features(df, numeric_cols, categorical_cols)
    y = df["click"]

    # ----- 2. VIF ------------------------------------------------------
    print("\n=== [2] VIF on the encoded design matrix ===")
    vif = compute_vif(X)
    print(vif.round(2).to_string())

    # ----- 4. logistic regression --------------------------------------
    print("\n=== [4] logistic regression, z-scored numerics (adjusted effects) ===")
    logit_res, logit_tab = fit_logistic_inference(X, y)
    with pd.option_context("display.float_format", fmt):
        print(logit_tab)

    # ----- 5. GBM + SHAP ----------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=0)
    model, auc_te = fit_gbm(X_tr, y_tr, X_te, y_te)
    print(f"\n=== [5] LightGBM test AUC = {auc_te:.4f} ===")
    sv = shap_values_of(model, X_te)
    shap_mean = pd.Series(np.abs(sv).mean(axis=0), index=X_te.columns,
                          name="mean_abs_shap").sort_values(ascending=False)
    print(shap_mean.round(4).to_string())

    shap.summary_plot(sv, X_te, show=False, max_display=12)
    plt.tight_layout(); plt.savefig("shap_beeswarm.png", dpi=120); plt.close()
    shap.dependence_plot("f_novelty", sv, X_te, interaction_index=None, show=False)
    plt.tight_layout(); plt.savefig("shap_dependence_novelty.png", dpi=120); plt.close()
    shap.dependence_plot("f_delivery_days", sv, X_te,
                         interaction_index="device_mobile", show=False)
    plt.tight_layout(); plt.savefig("shap_dependence_delivery.png", dpi=120); plt.close()

    print("\n--- top SHAP interaction pairs (1200-row subsample) ---")
    inter = shap_interaction_ranking(model, X_te.iloc[:1200])
    with pd.option_context("display.float_format", fmt):
        print(inter)

    # ----- 6. permutation importance: single vs cluster ----------------
    single_groups = {c: cols for c, cols in mapping.items()}          # per original feature
    clust_groups = {f"cluster_{cid}({'+'.join(clusters.index[clusters == cid])})":
                    sum((mapping[f] for f in clusters.index[clusters == cid]), [])
                    for cid in sorted(clusters.unique())}
    base_auc, perm_single = permutation_importance_auc(model, X_te, y_te, single_groups)
    _, perm_cluster = permutation_importance_auc(model, X_te, y_te, clust_groups)
    print("\n=== [6] permutation AUC drop -- single features ===")
    print(perm_single.round(4).to_string())
    print("\n--- permutation AUC drop -- clusters shuffled jointly ---")
    print(perm_cluster.round(4).to_string())

    # ----- 7. drop-cluster retrain -------------------------------------
    _, dropped = drop_cluster_importance(X_tr, y_tr, X_te, y_te, clust_groups)
    print("\n=== [7] AUC loss when a cluster is REMOVED and model retrained ===")
    print(dropped.round(4).to_string())

    # ----- 8. interaction LRTs ----------------------------------------
    print("\n=== [8] likelihood-ratio interaction tests ===")
    for fa, fb in [("f_delivery_days", "device"), ("f_price", "f_rating")]:
        r = interaction_lrt(df, "click", fa, fb, categorical_cols)
        print(f"  {fa} x {fb}: LRT={r['LRT']:.1f}, dof={r['dof']}, p={r['p_interaction']:.2e}")

    # ----- 9. ridge vs lasso on the collinear pair ---------------------
    print("\n=== [9] ridge vs lasso on [f_price, f_price_dup] only ===")
    print(regularization_tug_of_war_demo(X[["f_price", "f_price_dup"]], y).round(3))

    # ----- summary comparison table ------------------------------------
    uni = quick_univariate(df, numeric_cols, "click")
    summary = pd.DataFrame(index=numeric_cols + categorical_cols)
    summary["cohens_d"] = uni["cohens_d"]
    summary["auc_uni"] = uni["auc_uni"]
    summary["logit_p_min"] = [logit_tab.loc[mapping[f], "p"].min() for f in summary.index]
    summary["shap"] = [shap_mean[mapping[f]].sum() for f in summary.index]
    summary["perm_drop"] = [perm_single[f] for f in summary.index]
    summary["cluster"] = clusters
    print("\n=== SUMMARY: univariate vs adjusted vs model-based ===")
    with pd.option_context("display.float_format", fmt):
        print(summary)
    print("\nfigures saved: assoc_heatmap.png, feature_dendrogram.png, "
          "shap_beeswarm.png, shap_dependence_novelty.png, shap_dependence_delivery.png")
