"""
win_curve_context_selection.py
==============================

PROBLEM
-------
We have a bidding log:

    df[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'bid', 'bid_won']]

where 'bid_won' is 0/1.  We want to pick the *best subset of feature columns*
to condition on ("contextualize"), so that within each context the win curve

        P(bid_won = 1 | bid, context)

is estimated as accurately as the data can support.

DATA ASSUMPTIONS (important -- this version does NO binning)
------------------------------------------------------------
The data is already pre-processed:

    * every candidate feature has SMALL CARDINALITY (a handful of levels),
      so a context is simply the tuple of raw feature values;
    * `bid` takes only a SMALL NUMBER of distinct price points (say <= ~10),
      so every distinct bid value is its own "bid level" -- no bucketing.

A fail-fast guard (`validate_discreteness`) raises a clear error if a column
unexpectedly has high cardinality, because grouping on a continuous column
would silently explode the number of contexts.

METHOD (three ingredients, wired together by a greedy search)
-------------------------------------------------------------
1.  MODEL       Per-context ISOTONIC REGRESSION of `bid_won` on `bid`
                (win probability is non-decreasing in bid), with a GLOBAL
                isotonic curve as a fallback for contexts that are too small
                in a training fold or unseen at validation time.

2.  ACCURACY    K-fold CROSS-VALIDATED LOG-LOSS.  Log-loss is a *proper
                scoring rule*: it is minimized only by the true conditional
                probabilities, so it rewards calibration (what a win curve
                needs), not just ranking.  Overly fine partitions produce
                noisy curves and are automatically punished on held-out folds.

3.  SUFFICIENCY For every (context x bid level) cell we compute a
                WILSON (or CLOPPER-PEARSON) confidence interval for the
                empirical win rate.  A candidate subset is REJECTED OUTRIGHT
                when the MEDIAN CI WIDTH across cells exceeds a threshold:
                the partition is then too fine for the data to yield a
                trustworthy curve, regardless of its CV score.
                Empty cells (a context never observed at some bid price)
                optionally count with width = 1.0, penalizing coverage gaps.

4.  SEARCH      EXHAUSTIVE SUBSET SEARCH.  Every subset of the candidate
                features (2^k of them, size-capped by max_subset_size) is
                enumerated, INCLUDING the empty subset -- one GLOBAL curve --
                which is kept as the guaranteed fallback.  Each subset is
                screened by the sufficiency gate first; only gate-passing
                subsets are scored with cross-validation.  The winner is
                chosen by CV log-loss with a PARSIMONY rule: among subsets
                within `min_improvement` of the best score, the smallest one
                is returned, so extra features must earn their contexts by a
                clear margin rather than by fold noise.  Interaction effects
                (features useless alone, useful jointly) are captured by
                construction, since every combination is tried.

USAGE
-----
    from win_curve_context_selection import Config, run_selection, win_curve_table

    cfg    = Config()                      # tweak knobs as needed
    result = run_selection(df, cfg)        # df must contain the columns above
    result.summary()                       # human-readable report

    curves = win_curve_table(df, result.best_subset, cfg)   # final curve + CIs

Run this file directly (``python win_curve_context_selection.py``) to execute
a demo on synthetic *discrete* data whose ground truth depends only on
'a', 'b' and bid.

NOTES
-----
* If your data is TEMPORAL (it usually is in bidding), replace
  StratifiedKFold with time-ordered splits (e.g. sklearn TimeSeriesSplit) --
  see the comment inside `cv_log_loss_for_subset`.
* Isotonic regression still consumes the numeric bid values directly; the
  discrete bid levels are used for the sufficiency grid and for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


# ============================================================================
# 1. CONFIGURATION -- every tunable knob lives here, with its rationale
# ============================================================================

@dataclass
class Config:
    # ---- column layout -----------------------------------------------------
    candidate_features: Tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g")
    bid_col: str = "bid"
    target_col: str = "bid_won"

    # ---- discreteness guard ------------------------------------------------
    # The module assumes pre-discretized data.  If any candidate feature or
    # the bid column has more distinct values than this, we raise instead of
    # silently creating an astronomical number of contexts / bid levels.
    max_expected_cardinality: int = 50

    # ---- cross-validation --------------------------------------------------
    n_folds: int = 5
    random_state: int = 42
    # Minimum training rows a context needs (within a fold) to get its own
    # isotonic curve; smaller contexts fall back to the global curve.
    min_context_train_rows: int = 200
    # Probabilities are clipped to [eps, 1-eps] before log-loss so that a
    # single overconfident 0/1 prediction cannot produce an infinite loss.
    prob_clip: float = 1e-6

    # ---- sufficiency gate (Wilson / Clopper-Pearson) ------------------------
    ci_method: str = "wilson"           # "wilson" or "clopper_pearson"
    ci_alpha: float = 0.05              # 0.05 -> 95% confidence intervals
    # REJECTION RULE: a subset is rejected when the MEDIAN CI width over all
    # (context x bid level) cells exceeds this threshold.  E.g. 0.30 means
    # "in at least half of the cells the win rate is pinned down to +/-0.15".
    max_median_ci_width: float = 0.30
    # If True, cells of the full (context x bid level) grid that contain ZERO
    # observations enter the median with width 1.0 (maximum uncertainty).
    # This penalizes partitions whose contexts never see whole bid ranges.
    penalize_missing_cells: bool = True

    # ---- exhaustive search ---------------------------------------------------
    # PARSIMONY TOLERANCE: among gate-passing subsets whose CV log-loss lies
    # within `min_improvement` of the best score, the SMALLEST subset wins
    # (ties broken by lower loss, then lexicographically).  This counters the
    # "winner's curse" of exhaustive search -- with 2^k comparisons some big
    # subset will beat the truth by fold noise alone -- and it is what makes
    # the empty subset a real backup: extra features must earn their keep by
    # a clear margin, or the simpler/global partition is returned.
    min_improvement: float = 0.002
    # Optional hard cap on subset size (None = up to all candidates).  Also
    # the practical lever against exponential cost: with a cap of m the
    # number of enumerated subsets drops from 2^k to sum_{r<=m} C(k, r).
    max_subset_size: Optional[int] = None
    # Safety valve: refuse to enumerate absurdly many subsets.  7 features
    # -> 128 subsets is fine; 15 features -> 32768 is probably not.
    max_total_subsets: int = 4096


# ============================================================================
# 2. CONFIDENCE INTERVALS FOR A BINOMIAL PROPORTION (vectorized)
# ============================================================================

def wilson_interval(wins: np.ndarray, n: np.ndarray, alpha: float = 0.05
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wilson score interval for a binomial proportion.

    Preferred default: unlike the naive Wald interval it behaves sensibly for
    small n and for win rates near 0 or 1 (both common in bidding data at the
    extremes of the bid range).

    Parameters are arrays of successes (`wins`) and trials (`n`).
    Cells with n == 0 return the maximally uninformative interval [0, 1].
    """
    wins = np.asarray(wins, dtype=float)
    n = np.asarray(n, dtype=float)
    z = stats.norm.ppf(1.0 - alpha / 2.0)          # e.g. 1.96 for alpha=0.05

    with np.errstate(divide="ignore", invalid="ignore"):
        p_hat = np.where(n > 0, wins / n, np.nan)  # empirical win rate
        denom = 1.0 + z**2 / n
        centre = (p_hat + z**2 / (2.0 * n)) / denom
        half = (z * np.sqrt(p_hat * (1.0 - p_hat) / n
                            + z**2 / (4.0 * n**2))) / denom

    lo = np.clip(centre - half, 0.0, 1.0)
    hi = np.clip(centre + half, 0.0, 1.0)
    # Empty cells: no information at all -> [0, 1], i.e. width 1.
    lo = np.where(n > 0, lo, 0.0)
    hi = np.where(n > 0, hi, 1.0)
    return lo, hi


def clopper_pearson_interval(wins: np.ndarray, n: np.ndarray,
                             alpha: float = 0.05
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clopper-Pearson ("exact") interval, based on Beta-distribution quantiles.

    Guaranteed >= nominal coverage, hence conservative: intervals are wider
    than Wilson's, so with the same `max_median_ci_width` threshold the
    sufficiency gate becomes stricter.
    """
    wins = np.asarray(wins, dtype=float)
    n = np.asarray(n, dtype=float)

    with np.errstate(invalid="ignore"):
        # Standard CP construction; the np.where guards handle the edge cases
        # k == 0 (lower bound is exactly 0) and k == n (upper bound exactly 1),
        # where the Beta quantile would be undefined.
        lo = np.where(wins > 0,
                      stats.beta.ppf(alpha / 2.0, wins, n - wins + 1.0), 0.0)
        hi = np.where(wins < n,
                      stats.beta.ppf(1.0 - alpha / 2.0, wins + 1.0, n - wins),
                      1.0)

    lo = np.where(n > 0, np.nan_to_num(lo, nan=0.0), 0.0)
    hi = np.where(n > 0, np.nan_to_num(hi, nan=1.0), 1.0)
    return lo, hi


def _interval(wins, n, cfg: Config):
    """Dispatch to the interval method chosen in the config."""
    if cfg.ci_method == "wilson":
        return wilson_interval(wins, n, cfg.ci_alpha)
    if cfg.ci_method == "clopper_pearson":
        return clopper_pearson_interval(wins, n, cfg.ci_alpha)
    raise ValueError(f"Unknown ci_method: {cfg.ci_method!r}")


# ============================================================================
# 3. CONTEXT KEYS AND BID LEVELS (data is assumed pre-discretized -- no binning)
# ============================================================================

def validate_discreteness(df: pd.DataFrame, cfg: Config) -> None:
    """
    Fail fast if the data violates the "already discretized" assumption.

    Grouping on an accidentally-continuous column would create ~one context
    per row: sufficiency would (correctly) reject everything, but only after
    a lot of wasted compute and a confusing report.  Better to raise here
    with an actionable message.
    """
    for col in list(cfg.candidate_features) + [cfg.bid_col]:
        k = df[col].nunique()
        if k > cfg.max_expected_cardinality:
            raise ValueError(
                f"Column '{col}' has {k} distinct values, which exceeds "
                f"max_expected_cardinality={cfg.max_expected_cardinality}. "
                f"This module assumes pre-discretized features and a small "
                f"set of bid price points; discretize '{col}' upstream or "
                f"raise the limit if this is intentional."
            )


def stringify_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Verbatim string copies of every candidate feature.

    No quantile binning happens here (the data is already discrete); the
    string cast only exists so heterogeneous dtypes (ints, categories, ...)
    can be joined into a single context key.  Done ONCE up front so every
    subset evaluation reuses the same representation.
    """
    out = pd.DataFrame(index=df.index)
    for col in cfg.candidate_features:
        out[col] = df[col].astype(str)
    return out


def make_context_key(feat_str: pd.DataFrame,
                     subset: Tuple[str, ...]) -> pd.Series:
    """
    Collapse the chosen feature columns into one string key per row,
    e.g.  subset=('a','b')  ->  'seg_low|2'.

    Implemented as vectorized elementwise concatenation: the naive
    ``.agg("|".join, axis=1)`` runs a Python call per row (~2.4 s for 7
    features x 40k rows) which, multiplied by 2^k subsets, dominates the
    whole exhaustive search; this version is ~50x faster.

    The empty subset maps every row to the single context 'GLOBAL', which is
    exactly the "no contextualization" baseline.
    """
    if len(subset) == 0:
        return pd.Series("GLOBAL", index=feat_str.index)
    key = feat_str[subset[0]]
    for col in subset[1:]:
        key = key + "|" + feat_str[col]
    return key


def assign_bid_levels(bid: pd.Series) -> pd.Series:
    """
    Each DISTINCT bid price is its own level (no bucketing needed: the bid
    column only takes a small number of values, e.g. up to ~10 price points).

    Returned as an *ordered categorical* whose categories are the sorted
    unique prices; the sufficiency check uses `.cat.categories` to build the
    full (context x bid level) grid, so a context that never sees some price
    still produces an (empty) cell there.
    """
    levels = np.sort(bid.unique())
    return pd.Series(pd.Categorical(bid, categories=levels, ordered=True),
                     index=bid.index)


# ============================================================================
# 4. SUFFICIENCY GATE: median CI width over (context x bid level) cells
# ============================================================================

@dataclass
class SufficiencyReport:
    subset: Tuple[str, ...]
    passes: bool                    # median width <= cfg.max_median_ci_width ?
    median_ci_width: float          # THE rejection statistic
    traffic_weighted_width: float   # extra diagnostic (weights = cell size)
    n_contexts: int                 # how many contexts the subset induces
    n_cells: int                    # contexts x bid levels actually scored
    frac_empty_cells: float         # coverage gaps in the full grid
    cell_table: pd.DataFrame        # per-cell n / wins / rate / CI


def sufficiency_check(y: pd.Series, ctx: pd.Series, bid_level: pd.Series,
                      cfg: Config, subset: Tuple[str, ...]
                      ) -> SufficiencyReport:
    """
    Decide whether a candidate subset yields a partition the data can support.

    We compute, for every (context, bid level) cell, the CI of the empirical
    win rate, then REJECT the subset if the MEDIAN CI width exceeds
    `cfg.max_median_ci_width`.

    Computed on the FULL dataset (not per fold) on purpose: sufficiency is a
    property of the partition we would ship to production (fit on all data),
    while generalization is separately measured by CV log-loss.
    """
    cells = pd.DataFrame({
        "ctx": ctx.to_numpy(),
        "bid": bid_level.to_numpy(),       # raw price values; groupable
        "y": y.to_numpy(),
    })

    # Per-cell counts: n = auctions in the cell, wins = won auctions.
    agg = (cells.groupby(["ctx", "bid"], observed=True)["y"]
                .agg(n="size", wins="sum"))

    # Optionally expand to the FULL grid (every context x every bid price) so
    # that prices a context never sees count as width-1.0 cells.
    if cfg.penalize_missing_cells:
        full_grid = pd.MultiIndex.from_product(
            [np.unique(cells["ctx"]), list(bid_level.cat.categories)],
            names=["ctx", "bid"],
        )
        agg = agg.reindex(full_grid, fill_value=0)

    n = agg["n"].to_numpy(dtype=float)
    wins = agg["wins"].to_numpy(dtype=float)

    lo, hi = _interval(wins, n, cfg)
    width = hi - lo                                  # empty cells -> 1.0

    median_width = float(np.median(width))
    # Traffic-weighted mean width: "how uncertain is the curve for the
    # average auction?" (empty cells naturally drop out, weight 0).
    populated = n > 0
    weighted_width = (float(np.average(width[populated], weights=n[populated]))
                      if populated.any() else 1.0)

    table = agg.reset_index()
    table["win_rate"] = np.where(n > 0, wins / np.maximum(n, 1), np.nan)
    table["ci_lo"], table["ci_hi"], table["ci_width"] = lo, hi, width

    return SufficiencyReport(
        subset=subset,
        passes=median_width <= cfg.max_median_ci_width,
        median_ci_width=median_width,
        traffic_weighted_width=weighted_width,
        n_contexts=int(cells["ctx"].nunique()),
        n_cells=int(len(agg)),
        frac_empty_cells=float(np.mean(n == 0)),
        cell_table=table,
    )


# ============================================================================
# 5. ACCURACY: cross-validated log-loss of per-context isotonic curves
# ============================================================================

def _fit_isotonic(bids: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """
    One monotone win curve: P(win | bid) non-decreasing in bid.

    Works directly on the (few, discrete) numeric bid values -- isotonic
    regression does not care that x takes only ~10 distinct points.

    - y_min/y_max clamp outputs to valid probabilities;
    - out_of_bounds='clip' extends the curve flat beyond the observed bid
      range instead of raising on unseen bids at validation time.
    (Flip `increasing=False` if in your encoding a lower bid should win more.)
    """
    return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                              out_of_bounds="clip").fit(bids, y)


@dataclass
class CVResult:
    subset: Tuple[str, ...]
    mean_log_loss: float
    std_log_loss: float
    fold_log_losses: List[float]
    # Share of validation rows scored by their OWN context model (the rest
    # fell back to the global curve).  Low values reveal fragmentation even
    # before the sufficiency gate does.
    frac_context_scored: float


def cv_log_loss_for_subset(df: pd.DataFrame, feat_str: pd.DataFrame,
                           subset: Tuple[str, ...], cfg: Config,
                           folds: Optional[List[Tuple[np.ndarray,
                                                      np.ndarray]]] = None
                           ) -> CVResult:
    """
    K-fold CV estimate of the log-loss achieved by "group by `subset`,
    fit isotonic per context, fall back to global isotonic when small/unseen".

    Log-loss is a proper scoring rule, so this single number captures the
    bias-variance trade-off of the partition:
      * too coarse  -> heterogeneous contexts -> biased curves -> high loss;
      * too fine    -> tiny contexts -> noisy curves -> high loss on held-out.

    `folds` (optional): precomputed (train_idx, val_idx) pairs.  The
    exhaustive search passes the SAME folds to every subset -- identical
    splits make the 2^k loss comparisons exact and skip 2^k re-splits.
    """
    y_all = df[cfg.target_col].to_numpy()
    bids = df[cfg.bid_col].to_numpy(dtype=float)
    ctx = make_context_key(feat_str, subset).to_numpy()

    if folds is None:
        # NOTE (temporal data): bidding logs usually drift over time.  For a
        # production system replace StratifiedKFold with time-ordered splits,
        # e.g. sklearn.model_selection.TimeSeriesSplit, keeping the rest
        # as-is.
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                              random_state=cfg.random_state)
        folds = list(skf.split(np.zeros(len(df)), y_all))

    losses: List[float] = []
    ctx_scored: List[float] = []

    for train_idx, val_idx in folds:
        # ---- fit: one global fallback + one model per big-enough context ----
        global_model = _fit_isotonic(bids[train_idx], y_all[train_idx])

        train_frame = pd.DataFrame({"ctx": ctx[train_idx],
                                    "bid": bids[train_idx],
                                    "y": y_all[train_idx]})
        models: Dict[str, IsotonicRegression] = {}
        for c, g in train_frame.groupby("ctx", sort=False):
            # A context earns its own curve only with enough rows AND at
            # least two distinct bid values (otherwise no curve to speak of).
            # Contexts with constant y still get a curve (a constant one) --
            # if that constant is overconfident, CV log-loss will punish it,
            # which is precisely the selection mechanism at work.
            if len(g) >= cfg.min_context_train_rows and g["bid"].nunique() >= 2:
                models[c] = _fit_isotonic(g["bid"].to_numpy(),
                                          g["y"].to_numpy())

        # ---- predict on the validation fold, context by context -------------
        preds = np.empty(len(val_idx), dtype=float)
        val_frame = pd.DataFrame({"ctx": ctx[val_idx], "bid": bids[val_idx]},
                                 index=np.arange(len(val_idx)))
        n_context_scored = 0
        for c, g in val_frame.groupby("ctx", sort=False):
            model = models.get(c)
            if model is None:                 # small or unseen context
                model = global_model
            else:
                n_context_scored += len(g)
            preds[g.index] = model.predict(g["bid"].to_numpy())

        # ---- score -----------------------------------------------------------
        preds = np.clip(preds, cfg.prob_clip, 1.0 - cfg.prob_clip)
        losses.append(log_loss(y_all[val_idx], preds, labels=[0, 1]))
        ctx_scored.append(n_context_scored / len(val_idx))

    return CVResult(subset=subset,
                    mean_log_loss=float(np.mean(losses)),
                    std_log_loss=float(np.std(losses)),
                    fold_log_losses=[float(l) for l in losses],
                    frac_context_scored=float(np.mean(ctx_scored)))


# ============================================================================
# 6. GREEDY FORWARD SELECTION (sufficiency gate first, then CV score)
# ============================================================================

@dataclass
class SelectionResult:
    best_subset: Tuple[str, ...]
    best_log_loss: float
    baseline_log_loss: float        # empty subset = single global curve
    history: List[dict] = field(default_factory=list)
    config: Optional[Config] = None

    def summary(self, max_rows: Optional[int] = 25) -> pd.DataFrame:
        """
        Pretty-print the search and return the FULL audit table.

        The printed table is sorted by CV log-loss (gate-rejected subsets,
        which have no loss, sink to the bottom).  Exhaustive runs can produce
        100+ rows, so only the top `max_rows` are printed by default; pass
        max_rows=None to print everything.  The returned DataFrame is always
        complete and in evaluation order.
        """
        table = pd.DataFrame(self.history)
        n_rejected = int((table["action"] == "rejected (CI width)").sum())
        n_scored = int(table["action"].isin(
            ["evaluated", "fallback only (gate failed)"]).sum())
        print("\n================ SELECTION SUMMARY ================")
        print(f"Subsets CV-scored / gate-rejected   : "
              f"{n_scored} / {n_rejected}")
        print(f"Baseline (global curve) CV log-loss : "
              f"{self.baseline_log_loss:.5f}")
        print(f"Best subset                         : "
              f"{_fmt_subset(self.best_subset)}")
        print(f"Best CV log-loss                    : "
              f"{self.best_log_loss:.5f}")
        gain = self.baseline_log_loss - self.best_log_loss
        print(f"Improvement over baseline           : {gain:.5f}")
        print("---------------------------------------------------")
        shown = table.sort_values(["cv_log_loss", "size"],
                                  na_position="last")
        if max_rows is not None and len(shown) > max_rows:
            shown = shown.head(max_rows)
            print(f"(top {max_rows} of {len(table)} rows by CV log-loss; "
                  f"call summary(max_rows=None) for the full table)")
        with pd.option_context("display.width", 160,
                               "display.max_columns", None):
            print(shown.to_string(index=False,
                                  float_format=lambda v: f"{v:.5f}"))
        print("===================================================\n")
        return table


def _fmt_subset(subset: Tuple[str, ...]) -> str:
    return "{" + ", ".join(subset) + "}" if subset else "{} (global)"


def run_selection(df: pd.DataFrame, cfg: Config = Config(),
                  verbose: bool = True) -> SelectionResult:
    """
    Exhaustive subset search:

        1. enumerate ALL subsets of cfg.candidate_features (sizes 0 up to
           max_subset_size), starting with the EMPTY subset -- the single
           GLOBAL curve, which doubles as the guaranteed fallback;
        2. for each subset run the SUFFICIENCY GATE first (cheap) -- subsets
           whose median (context x bid level) CI width exceeds the threshold
           are rejected WITHOUT spending CV compute on them;
        3. score every gate-passing subset with K-fold CV log-loss;
        4. pick the winner by loss + PARSIMONY: with the best raw score L*,
           keep every subset whose loss is <= L* + cfg.min_improvement, and
           among those return the SMALLEST one (ties -> lower loss -> lexico-
           graphic).  A bigger subset must therefore beat the smaller ones by
           a clear margin, not by fold noise -- and the global subset wins
           whenever nothing beats it decisively ("global as backup").

    Interaction effects need no special handling here: {a,b,c,d} is scored
    whether or not {a,b,c} helps on its own.  The price is exponential cost
    (2^k subsets), kept in check by the sufficiency gate, max_subset_size,
    and the max_total_subsets safety valve.
    """
    # ---- guard + one-off preprocessing shared by every candidate subset -----
    validate_discreteness(df, cfg)                 # fail fast on wrong input
    feat_str = stringify_features(df, cfg)         # verbatim, no binning
    bid_levels = assign_bid_levels(df[cfg.bid_col])
    y = df[cfg.target_col]
    max_size = cfg.max_subset_size or len(cfg.candidate_features)

    # ---- enumerate the powerset (size-ordered, deterministic) ----------------
    all_subsets: List[Tuple[str, ...]] = [
        subset
        for size in range(0, max_size + 1)
        for subset in combinations(cfg.candidate_features, size)
    ]
    if len(all_subsets) > cfg.max_total_subsets:
        raise ValueError(
            f"Exhaustive search would evaluate {len(all_subsets)} subsets, "
            f"exceeding max_total_subsets={cfg.max_total_subsets}.  The cost "
            f"is exponential in the number of features: trim "
            f"candidate_features, lower max_subset_size, or raise the limit "
            f"if this is intentional."
        )
    if verbose:
        print(f"Enumerating {len(all_subsets)} subsets of "
              f"{list(cfg.candidate_features)} (max size {max_size}).")

    # One set of folds shared by EVERY subset: identical splits make the
    # 2^k loss comparisons exact and avoid re-splitting per subset.
    # (Temporal data: swap in time-ordered splits here, see CV docstring.)
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                          random_state=cfg.random_state)
    folds = list(skf.split(np.zeros(len(df)), y.to_numpy()))

    history: List[dict] = []

    def record(action: str, subset: Tuple[str, ...],
               suff: SufficiencyReport, cv: Optional[CVResult]) -> None:
        """Append one line of the audit trail (kept for the final report)."""
        history.append({
            "size": len(subset),
            "action": action,                       # evaluated / rejected /
            "subset": _fmt_subset(subset),          #   selected / fallback
            "n_contexts": suff.n_contexts,
            "median_ci_width": suff.median_ci_width,
            "frac_empty_cells": suff.frac_empty_cells,
            "cv_log_loss": cv.mean_log_loss if cv else np.nan,
            "cv_std": cv.std_log_loss if cv else np.nan,
            "frac_context_scored": cv.frac_context_scored if cv else np.nan,
        })

    # ---- evaluate every subset: gate first, then CV --------------------------
    scored: List[Tuple[CVResult, SufficiencyReport]] = []
    base_cv: Optional[CVResult] = None             # CV of the empty subset

    for subset in all_subsets:
        ctx = make_context_key(feat_str, subset)

        # (1) sufficiency gate -- reject before paying for CV.  The EMPTY
        # subset is exempt from rejection: it is the designated backup and is
        # always CV-scored, even in the pathological case of a threshold so
        # tight that the single global curve fails it.
        suff = sufficiency_check(y, ctx, bid_levels, cfg, subset)
        if not suff.passes and subset != ():
            record("rejected (CI width)", subset, suff, None)
            if verbose:
                print(f"[size {len(subset)}] REJECT {_fmt_subset(subset)} : "
                      f"median CI width {suff.median_ci_width:.3f} "
                      f"> {cfg.max_median_ci_width} "
                      f"({suff.n_contexts} contexts, "
                      f"{suff.frac_empty_cells:.0%} empty cells)")
            continue

        # (2) accuracy -- CV log-loss of the per-context isotonic model.
        cv = cv_log_loss_for_subset(df, feat_str, subset, cfg, folds=folds)
        if subset == ():
            base_cv = cv
        if suff.passes:
            scored.append((cv, suff))
            record("evaluated", subset, suff, cv)
        else:                                       # empty subset, gate failed
            record("fallback only (gate failed)", subset, suff, cv)
        if verbose:
            print(f"[size {len(subset)}] eval   {_fmt_subset(subset)} : "
                  f"CV log-loss = {cv.mean_log_loss:.5f} "
                  f"(+/-{cv.std_log_loss:.5f}), "
                  f"median CI width = {suff.median_ci_width:.3f}, "
                  f"context-scored rows = {cv.frac_context_scored:.0%}")

    # ---- pick the winner: best loss, then parsimony --------------------------
    if scored:
        best_raw = min(cv.mean_log_loss for cv, _ in scored)
        # Subsets statistically indistinguishable from the best raw score.
        # With 2^k comparisons the raw argmin is biased toward big subsets
        # ("winner's curse"), so extra features must earn their keep by more
        # than the tolerance -- otherwise the smallest contender (possibly
        # the global one) is returned.
        contenders = [(cv, s) for cv, s in scored
                      if cv.mean_log_loss <= best_raw + cfg.min_improvement]
        cv_sel, suff_sel = min(
            contenders,
            key=lambda t: (len(t[0].subset), t[0].mean_log_loss, t[0].subset))
        record("selected", cv_sel.subset, suff_sel, cv_sel)
        if verbose:
            print(f"SELECTED {_fmt_subset(cv_sel.subset)} : "
                  f"CV log-loss = {cv_sel.mean_log_loss:.5f} "
                  f"(best raw = {best_raw:.5f}, parsimony tolerance = "
                  f"{cfg.min_improvement}, contenders = {len(contenders)})")
        best_subset, best_loss = cv_sel.subset, cv_sel.mean_log_loss
    else:
        # Nothing passed the gate at all -> fall back to the global curve.
        best_subset, best_loss = (), base_cv.mean_log_loss
        if verbose:
            print(f"No subset passed the sufficiency gate -> falling back "
                  f"to {_fmt_subset(())} with CV log-loss "
                  f"{base_cv.mean_log_loss:.5f}.")

    return SelectionResult(best_subset=best_subset, best_log_loss=best_loss,
                           baseline_log_loss=base_cv.mean_log_loss,
                           history=history, config=cfg)


# ============================================================================
# 7. FINAL DELIVERABLE: win curve per context, with CIs per bid price
# ============================================================================

def win_curve_table(df: pd.DataFrame, subset: Tuple[str, ...],
                    cfg: Config = Config()) -> pd.DataFrame:
    """
    Fit the chosen partition on the FULL dataset and return, per
    (context, bid price):

        n, wins, empirical win_rate, ci_lo, ci_hi, ci_width,
        iso_win_rate  (smoothed isotonic estimate at that exact bid price)

    Because bids are discrete, each row IS one point of the win curve.
    This is the artifact you would plot / ship: the empirical points with
    their Wilson (or CP) error bars, plus the monotone isotonic curve.
    (Observed cells only; coverage gaps are already policed by the
    sufficiency gate during selection.)
    """
    validate_discreteness(df, cfg)
    feat_str = stringify_features(df, cfg)
    ctx = make_context_key(feat_str, subset)

    frame = pd.DataFrame({"ctx": ctx.to_numpy(),
                          "bid": df[cfg.bid_col].to_numpy(dtype=float),
                          "y": df[cfg.target_col].to_numpy()})

    # ---- empirical cells: one row per (context, distinct bid price) ---------
    agg = (frame.groupby(["ctx", "bid"])
                .agg(n=("y", "size"), wins=("y", "sum"))
                .reset_index())

    lo, hi = _interval(agg["wins"].to_numpy(float),
                       agg["n"].to_numpy(float), cfg)
    agg["win_rate"] = agg["wins"] / agg["n"]
    agg["ci_lo"], agg["ci_hi"] = lo, hi
    agg["ci_width"] = agg["ci_hi"] - agg["ci_lo"]

    # ---- isotonic curves fitted on ALL data (the production model) ----------
    global_model = _fit_isotonic(frame["bid"].to_numpy(),
                                 frame["y"].to_numpy())
    models: Dict[str, IsotonicRegression] = {}
    for c, g in frame.groupby("ctx", sort=False):
        if len(g) >= cfg.min_context_train_rows and g["bid"].nunique() >= 2:
            models[c] = _fit_isotonic(g["bid"].to_numpy(), g["y"].to_numpy())

    # Evaluate each context's curve at each of its observed bid prices.
    iso = np.empty(len(agg), dtype=float)
    for c, g in agg.groupby("ctx", sort=False):
        model = models.get(c, global_model)
        iso[g.index] = model.predict(g["bid"].to_numpy())
    agg["iso_win_rate"] = iso

    return agg.sort_values(["ctx", "bid"]).reset_index(drop=True)


# ============================================================================
# 8. DEMO on synthetic DISCRETE data (ground truth depends on 'a', 'b', bid)
# ============================================================================

def _make_synthetic(n: int = 40_000, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic bidding log matching the "already processed" assumptions:
    every feature is low-cardinality and bids take 10 discrete price points.

    KNOWN generating process:

        P(win | bid, a, b) = sigmoid( (bid - threshold(a) - 0.7*b) / 0.9 )

    so the correct contextualization is {a, b}; c..g are pure noise.
    The demo should (i) pick {a, b}, (ii) cheaply reject over-fine subsets
    via the CI-width gate, and (iii) prefer {a, b} over noise-padded
    supersets via CV log-loss + the parsimony rule.
    """
    rng = np.random.default_rng(seed)
    a = rng.choice(["seg_low", "seg_mid", "seg_high"], size=n, p=[.5, .3, .2])
    b = rng.integers(0, 4, size=n)                    # ordinal 0..3, REAL signal
    df = pd.DataFrame({
        "a": a,                                       # 3 levels, real signal
        "b": b,                                       # 4 levels, real signal
        "c": rng.choice(list("WXYZ"), size=n),        # noise, 4 levels
        "d": rng.choice([f"d{i}" for i in range(5)],  # noise, 5 levels
                        size=n),
        "e": rng.integers(0, 5, size=n),              # noise, 5 levels (ints)
        "f": rng.choice(["p", "q"], size=n),          # noise, 2 levels
        "g": rng.choice(["g_lo", "g_mid", "g_hi"],    # noise, 3 levels
                        size=n),
        # 10 discrete price points: 1.0, 2.0, ..., 10.0
        "bid": rng.choice(np.arange(1.0, 11.0), size=n),
    })
    threshold = df["a"].map({"seg_low": 2.5, "seg_mid": 4.5,
                             "seg_high": 6.5}).to_numpy()
    p_win = 1.0 / (1.0 + np.exp(-(df["bid"].to_numpy()
                                  - threshold - 0.7 * b) / 0.9))
    df["bid_won"] = rng.binomial(1, p_win)
    return df


if __name__ == "__main__":
    # ---- build data & configure --------------------------------------------
    df = _make_synthetic()

    # NOTE: max_median_ci_width is set deliberately tight here (0.10) so the
    # demo visibly exercises the rejection path; with real data something in
    # the 0.2-0.4 range is a more typical starting point (0.30 means: in at
    # least half of the cells the win rate is pinned down to +/- 0.15).
    cfg = Config(max_median_ci_width=0.10,   # sufficiency threshold
                 n_folds=5,
                 min_context_train_rows=200,
                 min_improvement=0.002)      # also the parsimony tolerance

    # ---- run the search ------------------------------------------------------
    result = run_selection(df, cfg, verbose=True)
    result.summary()

    # ---- inspect the final win curves ----------------------------------------
    curves = win_curve_table(df, result.best_subset, cfg)
    print("Final win-curve table (first bid levels of the first contexts):")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(curves.head(15).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
