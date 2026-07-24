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

3.  SUFFICIENCY For every (context x bid-bucket) cell we compute a
                WILSON (or CLOPPER-PEARSON) confidence interval for the
                empirical win rate.  A candidate subset is REJECTED OUTRIGHT
                when the MEDIAN CI WIDTH across cells exceeds a threshold:
                the partition is then too fine for the data to yield a
                trustworthy curve, regardless of its CV score.
                Empty cells (a context never observed in some bid range)
                optionally count with width = 1.0, penalizing coverage gaps.

4.  SEARCH      GREEDY FORWARD SELECTION.  The number of contexts explodes
                combinatorially with subset size, so we grow the subset one
                feature at a time, keeping the addition that most reduces
                CV log-loss, and stop when the improvement falls below a
                tolerance or every remaining candidate fails sufficiency.

USAGE
-----
    from win_curve_context_selection import Config, run_selection, win_curve_table

    cfg    = Config()                      # tweak knobs as needed
    result = run_selection(df, cfg)        # df must contain the columns above
    result.summary()                       # human-readable report

    curves = win_curve_table(df, result.best_subset, cfg)   # final curve + CIs

Run this file directly (``python win_curve_context_selection.py``) to execute
a demo on synthetic data whose ground truth depends only on 'a', 'b' and bid.

NOTES
-----
* If your data is TEMPORAL (it usually is in bidding), replace
  StratifiedKFold with time-ordered splits (e.g. sklearn TimeSeriesSplit) --
  see the comment inside `cv_log_loss_for_subset`.
* Continuous features are quantile-binned before grouping; categorical
  features are used as-is.  Contexts are simply the tuple of binned values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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

    # ---- discretization ----------------------------------------------------
    # Continuous features are quantile-binned into this many bins before
    # grouping (a context must be a *discrete* cell).  Features that are
    # already low-cardinality (nunique <= n_feature_bins) are used as-is.
    n_feature_bins: int = 5
    # Bid buckets are used ONLY for the sufficiency check and for reporting
    # the final curve; the isotonic model itself consumes the raw bid.
    n_bid_buckets: int = 10

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
    # (context x bid-bucket) cells exceeds this threshold.  E.g. 0.30 means
    # "in at least half of the cells the win rate is pinned down to +/-0.15".
    max_median_ci_width: float = 0.30
    # If True, cells of the full (context x bucket) grid that contain ZERO
    # observations enter the median with width 1.0 (maximum uncertainty).
    # This penalizes partitions whose contexts never see whole bid ranges.
    penalize_missing_cells: bool = True

    # ---- greedy search -----------------------------------------------------
    # Stop when the best candidate improves CV log-loss by less than this.
    min_improvement: float = 0.002
    # Optional hard cap on subset size (None = up to all candidates).
    max_subset_size: Optional[int] = None


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
# 3. DISCRETIZATION AND CONTEXT KEYS
# ============================================================================

def discretize_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Return a DataFrame of *string-coded* versions of every candidate feature.

    - numeric feature with many distinct values -> quantile bins (pd.qcut),
      so each bin holds roughly the same amount of traffic;
    - anything else (categoricals, low-cardinality ints) -> used verbatim.

    Doing this ONCE up front means every subset evaluation reuses the same
    binning, which keeps comparisons between subsets apples-to-apples.
    """
    binned = pd.DataFrame(index=df.index)
    for col in cfg.candidate_features:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > cfg.n_feature_bins:
            # duplicates="drop" tolerates heavy ties (fewer bins than asked).
            binned[col] = pd.qcut(s, q=cfg.n_feature_bins,
                                  duplicates="drop").astype(str)
        else:
            binned[col] = s.astype(str)
    return binned


def make_context_key(binned: pd.DataFrame,
                     subset: Tuple[str, ...]) -> pd.Series:
    """
    Collapse the chosen (binned) feature columns into one string key per row,
    e.g.  subset=('a','b')  ->  'seg_low|(0.2, 0.4]'.

    The empty subset maps every row to the single context 'GLOBAL', which is
    exactly the "no contextualization" baseline.
    """
    if len(subset) == 0:
        return pd.Series("GLOBAL", index=binned.index)
    return binned[list(subset)].agg("|".join, axis=1)


def assign_bid_buckets(bid: pd.Series, cfg: Config) -> pd.Series:
    """
    Global quantile buckets of the bid (categorical).  Defined once on the
    FULL dataset so that every subset's sufficiency check uses identical
    buckets -- otherwise thresholds would not be comparable across subsets.
    """
    return pd.qcut(bid, q=cfg.n_bid_buckets, duplicates="drop")


# ============================================================================
# 4. SUFFICIENCY GATE: median CI width over (context x bid-bucket) cells
# ============================================================================

@dataclass
class SufficiencyReport:
    subset: Tuple[str, ...]
    passes: bool                    # median width <= cfg.max_median_ci_width ?
    median_ci_width: float          # THE rejection statistic
    traffic_weighted_width: float   # extra diagnostic (weights = cell size)
    n_contexts: int                 # how many contexts the subset induces
    n_cells: int                    # contexts x buckets actually scored
    frac_empty_cells: float         # coverage gaps in the full grid
    cell_table: pd.DataFrame        # per-cell n / wins / rate / CI


def sufficiency_check(y: pd.Series, ctx: pd.Series, bid_bucket: pd.Series,
                      cfg: Config, subset: Tuple[str, ...]
                      ) -> SufficiencyReport:
    """
    Decide whether a candidate subset yields a partition the data can support.

    We compute, for every (context, bid-bucket) cell, the CI of the empirical
    win rate, then REJECT the subset if the MEDIAN CI width exceeds
    `cfg.max_median_ci_width`.

    Computed on the FULL dataset (not per fold) on purpose: sufficiency is a
    property of the partition we would ship to production (fit on all data),
    while generalization is separately measured by CV log-loss.
    """
    cells = pd.DataFrame({
        "ctx": ctx.to_numpy(),
        "bucket": bid_bucket.to_numpy(),   # Interval objects; groupable
        "y": y.to_numpy(),
    })

    # Per-cell counts: n = impressions in the cell, wins = won auctions.
    agg = (cells.groupby(["ctx", "bucket"], observed=True)["y"]
                .agg(n="size", wins="sum"))

    # Optionally expand to the FULL grid (every context x every bucket) so
    # that bid ranges a context never sees count as width-1.0 cells.
    if cfg.penalize_missing_cells:
        full_grid = pd.MultiIndex.from_product(
            [np.unique(cells["ctx"]), list(bid_bucket.cat.categories)],
            names=["ctx", "bucket"],
        )
        agg = agg.reindex(full_grid, fill_value=0)

    n = agg["n"].to_numpy(dtype=float)
    wins = agg["wins"].to_numpy(dtype=float)

    lo, hi = _interval(wins, n, cfg)
    width = hi - lo                                  # empty cells -> 1.0

    median_width = float(np.median(width))
    # Traffic-weighted mean width: "how uncertain is the curve for the
    # average impression?" (empty cells naturally drop out, weight 0).
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


def cv_log_loss_for_subset(df: pd.DataFrame, binned: pd.DataFrame,
                           subset: Tuple[str, ...], cfg: Config) -> CVResult:
    """
    K-fold CV estimate of the log-loss achieved by "group by `subset`,
    fit isotonic per context, fall back to global isotonic when small/unseen".

    Log-loss is a proper scoring rule, so this single number captures the
    bias-variance trade-off of the partition:
      * too coarse  -> heterogeneous contexts -> biased curves -> high loss;
      * too fine    -> tiny contexts -> noisy curves -> high loss on held-out.
    """
    y_all = df[cfg.target_col].to_numpy()
    bids = df[cfg.bid_col].to_numpy(dtype=float)
    ctx = make_context_key(binned, subset).to_numpy()

    # NOTE (temporal data): bidding logs usually drift over time.  For a
    # production system replace StratifiedKFold with time-ordered splits,
    # e.g. sklearn.model_selection.TimeSeriesSplit, keeping the rest as-is.
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                          random_state=cfg.random_state)

    losses: List[float] = []
    ctx_scored: List[float] = []

    for train_idx, val_idx in skf.split(np.zeros(len(df)), y_all):
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

    def summary(self) -> pd.DataFrame:
        """Pretty-print the whole search and return it as a DataFrame."""
        table = pd.DataFrame(self.history)
        print("\n================ SELECTION SUMMARY ================")
        print(f"Baseline (global curve) CV log-loss : "
              f"{self.baseline_log_loss:.5f}")
        print(f"Best subset                         : "
              f"{_fmt_subset(self.best_subset)}")
        print(f"Best CV log-loss                    : "
              f"{self.best_log_loss:.5f}")
        gain = self.baseline_log_loss - self.best_log_loss
        print(f"Improvement over baseline           : {gain:.5f}")
        print("---------------------------------------------------")
        with pd.option_context("display.width", 160,
                               "display.max_columns", None):
            print(table.to_string(index=False,
                                  float_format=lambda v: f"{v:.5f}"))
        print("===================================================\n")
        return table


def _fmt_subset(subset: Tuple[str, ...]) -> str:
    return "{" + ", ".join(subset) + "}" if subset else "{} (global)"


def run_selection(df: pd.DataFrame, cfg: Config = Config(),
                  verbose: bool = True) -> SelectionResult:
    """
    Greedy forward selection:

        step 0 : score the EMPTY subset (one global isotonic curve);
        step k : for every remaining feature f, form  best_subset + (f,);
                   * run the SUFFICIENCY GATE first (cheap) -- if the median
                     CI width exceeds the threshold, the candidate is
                     rejected WITHOUT spending CV compute on it;
                   * otherwise run K-fold CV and record the log-loss;
                 accept the candidate with the lowest CV log-loss if it
                 improves on the incumbent by > cfg.min_improvement;
        stop   : no candidate passes, or improvement is below tolerance,
                 or max_subset_size is reached.
    """
    # ---- one-off preprocessing shared by every candidate subset -------------
    binned = discretize_features(df, cfg)
    print(f'binned: {binned.shape}')
    print(binned.head())
    print('=================')
    bid_buckets = assign_bid_buckets(df[cfg.bid_col], cfg)
    print(f'bid_buckets: {bid_buckets.shape}')
    print(bid_buckets.head())
    print('=================')
    y = df[cfg.target_col]
    max_size = cfg.max_subset_size or len(cfg.candidate_features)

    history: List[dict] = []

    def record(step: int, action: str, subset: Tuple[str, ...],
               suff: SufficiencyReport, cv: Optional[CVResult]) -> None:
        """Append one line of the audit trail (kept for the final report)."""
        history.append({
            "step": step,
            "action": action,                       # baseline/evaluated/
            "subset": _fmt_subset(subset),          #   rejected/accepted
            "n_contexts": suff.n_contexts,
            "median_ci_width": suff.median_ci_width,
            "frac_empty_cells": suff.frac_empty_cells,
            "cv_log_loss": cv.mean_log_loss if cv else np.nan,
            "cv_std": cv.std_log_loss if cv else np.nan,
            "frac_context_scored": cv.frac_context_scored if cv else np.nan,
        })

    # ---- step 0: baseline = no contextualization ----------------------------
    base_subset: Tuple[str, ...] = ()
    base_suff = sufficiency_check(y, make_context_key(binned, base_subset),
                                  bid_buckets, cfg, base_subset)
    base_cv = cv_log_loss_for_subset(df, binned, base_subset, cfg)
    record(0, "baseline", base_subset, base_suff, base_cv)
    if verbose:
        print(f"[step 0] baseline {_fmt_subset(base_subset)} : "
              f"CV log-loss = {base_cv.mean_log_loss:.5f}, "
              f"median CI width = {base_suff.median_ci_width:.3f}")

    best_subset, best_loss = base_subset, base_cv.mean_log_loss
    remaining = list(cfg.candidate_features)
    step = 0

    # ---- greedy loop ---------------------------------------------------------
    while remaining and len(best_subset) < max_size:
        step += 1
        passing: List[Tuple[CVResult, SufficiencyReport]] = []

        for feat in remaining:
            candidate = best_subset + (feat,)
            ctx = make_context_key(binned, candidate)
            print(f'--------> ctx: {ctx.shape} {df.shape}, candidate: {candidate}')

            # (1) sufficiency gate -- reject before paying for CV.
            suff = sufficiency_check(y, ctx, bid_buckets, cfg, candidate)
            if not suff.passes:
                record(step, "rejected (CI width)", candidate, suff, None)
                if verbose:
                    print(f"[step {step}] REJECT {_fmt_subset(candidate)} : "
                          f"median CI width {suff.median_ci_width:.3f} "
                          f"> {cfg.max_median_ci_width} "
                          f"({suff.n_contexts} contexts, "
                          f"{suff.frac_empty_cells:.0%} empty cells)")
                continue

            # (2) accuracy -- CV log-loss of the per-context isotonic model.
            cv = cv_log_loss_for_subset(df, binned, candidate, cfg)
            record(step, "evaluated", candidate, suff, cv)
            passing.append((cv, suff))
            if verbose:
                print(f"[step {step}] eval   {_fmt_subset(candidate)} : "
                      f"CV log-loss = {cv.mean_log_loss:.5f} "
                      f"(+/-{cv.std_log_loss:.5f}), "
                      f"median CI width = {suff.median_ci_width:.3f}, "
                      f"context-scored rows = {cv.frac_context_scored:.0%}")

        if not passing:
            if verbose:
                print(f"[step {step}] every remaining candidate failed the "
                      f"sufficiency gate -> stop.")
            break

        # Best passing candidate of this step.
        cv_best, suff_best = min(passing, key=lambda t: t[0].mean_log_loss)
        improvement = best_loss - cv_best.mean_log_loss

        if improvement > cfg.min_improvement:
            best_subset = cv_best.subset
            best_loss = cv_best.mean_log_loss
            remaining.remove(best_subset[-1])         # consume the feature
            record(step, "accepted", best_subset, suff_best, cv_best)
            if verbose:
                print(f"[step {step}] ACCEPT {_fmt_subset(best_subset)} "
                      f"(improvement {improvement:.5f})")
        else:
            if verbose:
                print(f"[step {step}] best improvement {improvement:.5f} "
                      f"<= tolerance {cfg.min_improvement} -> stop.")
            break

    return SelectionResult(best_subset=best_subset, best_log_loss=best_loss,
                           baseline_log_loss=base_cv.mean_log_loss,
                           history=history, config=cfg)


# ============================================================================
# 7. FINAL DELIVERABLE: win curve per context, with CIs per bid bucket
# ============================================================================

def win_curve_table(df: pd.DataFrame, subset: Tuple[str, ...],
                    cfg: Config = Config()) -> pd.DataFrame:
    """
    Fit the chosen partition on the FULL dataset and return, per
    (context, bid-bucket):

        n, wins, empirical win_rate, ci_lo, ci_hi, ci_width,
        bid_median, iso_win_rate  (smoothed isotonic estimate at bid_median)

    This is the artifact you would plot / ship: the empirical points with
    their Wilson (or CP) error bars, plus the monotone isotonic curve.
    """
    binned = discretize_features(df, cfg)
    ctx = make_context_key(binned, subset)
    buckets = assign_bid_buckets(df[cfg.bid_col], cfg)

    frame = pd.DataFrame({"ctx": ctx.to_numpy(),
                          "bucket": buckets.to_numpy(),
                          "bid": df[cfg.bid_col].to_numpy(dtype=float),
                          "y": df[cfg.target_col].to_numpy()})

    # ---- empirical cells (observed only; reporting, not gating) -------------
    agg = (frame.groupby(["ctx", "bucket"], observed=True)
                .agg(n=("y", "size"), wins=("y", "sum"),
                     bid_median=("bid", "median"))
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

    # Evaluate each cell's curve at the cell's median bid.
    iso = np.empty(len(agg), dtype=float)
    for c, g in agg.groupby("ctx", sort=False):
        model = models.get(c, global_model)
        iso[g.index] = model.predict(g["bid_median"].to_numpy())
    agg["iso_win_rate"] = iso

    return agg.sort_values(["ctx", "bid_median"]).reset_index(drop=True)


# ============================================================================
# 8. DEMO on synthetic data (ground truth depends only on 'a', 'b' and bid)
# ============================================================================

def _make_synthetic(n: int = 40_000, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic bidding log with a KNOWN generating process:

        P(win | bid, a, b) = sigmoid( (bid - threshold(a) - 2*b) / 0.9 )

    so the correct contextualization is {a, b}; c..g are pure noise.
    The demo should (i) pick {a, b}, (ii) reject over-fine subsets via the
    CI-width gate, and (iii) stop adding noise features via CV log-loss.
    """
    rng = np.random.default_rng(seed)
    a = rng.choice(["seg_low", "seg_mid", "seg_high"], size=n, p=[.5, .3, .2])
    b = rng.uniform(0.0, 1.0, size=n)                 # continuous, real signal
    df = pd.DataFrame({
        "a": a,
        "b": b,
        "c": rng.choice(list("WXYZ"), size=n),        # noise, categorical
        "d": rng.normal(size=n),                      # noise, continuous
        "e": rng.integers(0, 5, size=n),              # noise, small-int
        "f": rng.choice(["p", "q"], size=n),          # noise, binary
        "g": rng.exponential(1.0, size=n),            # noise, continuous
        "bid": rng.uniform(0.0, 10.0, size=n),
    })
    threshold = df["a"].map({"seg_low": 2.5, "seg_mid": 4.5,
                             "seg_high": 6.5}).to_numpy()
    p_win = 1.0 / (1.0 + np.exp(-(df["bid"].to_numpy()
                                  - threshold - 2.0 * b) / 0.9))
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
                 n_feature_bins=5,
                 n_bid_buckets=10,
                 n_folds=5,
                 min_context_train_rows=200,
                 min_improvement=0.002)

    # ---- run the search ------------------------------------------------------
    result = run_selection(df, cfg, verbose=True)
    result.summary()

    # ---- inspect the final win curves ----------------------------------------
    curves = win_curve_table(df, result.best_subset, cfg)
    print("Final win-curve table (first 15 cells):")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(curves.head(15).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
