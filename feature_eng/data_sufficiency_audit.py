"""
data_sufficiency_audit.py
=========================
Answers: "Do I have enough data (rows, clicks, coverage) to train a click
classifier?" -- with a PASS / WARN / FAIL verdict and a plain-English reason
per check, plus an estimate of how many rows you actually need.

The six checks
--------------
1. PRECISION : is overall CTR known within the margin you want?
               n_required = 1.96^2 * (1-p) / (p * rel_margin^2)
2. EPV       : events (clicks) per model parameter >= 20 (10 = warning).
               n_required = min_epv * k / p
3. LEVELS    : does every categorical level have enough clicks (n*s*p)?
               + Good-Turing unseen-level mass = (#levels seen once) / n
4. RANGES    : is every part of the numeric prediction range supported by
               training rows (no extrapolation, no internal gaps)?
5. POWER     : can you detect the CTR differences you care about?
               n_per_group = 16 * p(1-p) / delta^2
6. LEARNING CURVE : train on growing fractions; rising = more rows help,
               flat above target AUC = enough, flat below = missing FEATURES.

Usage on your data
------------------
    from data_sufficiency_audit import audit, AuditConfig
    report, n_needed, verdict = audit(df, "click",
                                      numeric_cols, categorical_cols,
                                      pred_ranges={"price": (0, 500)},
                                      cfg=AuditConfig(target_auc=0.65))

Run this file directly for six synthetic scenarios, each built to
trigger one specific verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
@dataclass
class AuditConfig:
    target_rel_margin: float = 0.10   # want CTR known within +-10% (relative)
    min_epv: int = 20                 # clicks per parameter (10 = warn floor)
    min_clicks_per_level: int = 20    # per categorical level
    unseen_mass_max: float = 0.001    # Good-Turing: tolerated unseen-level share
    min_rows_per_range_bin: int = 50  # per decile of the prediction range
    target_delta_rel: float = 0.10    # want to detect a 10% relative CTR diff
    power_group_share: float = 0.5    # size of the smaller group compared
    target_auc: float = 0.65          # AUC you need from the model
    lc_fractions: tuple = (0.05, 0.10, 0.20, 0.40, 0.70, 1.00)
    lc_rise_eps: float = 0.004        # AUC gain over last step that counts as "rising"


def _res(check, status, measured, required, reason, n_req=None):
    return {"check": check, "status": status, "measured": measured,
            "required": required, "reason": reason, "n_req": n_req}


# ---------------------------------------------------------------- checks
def check_precision(y, cfg):
    n, p = len(y), y.mean()
    if p == 0:
        return _res("precision", "FAIL", "0 clicks", "-",
                    "No clicks at all: nothing can be estimated.", None)
    rel_m = 1.96 * np.sqrt((1 - p) / (n * p))          # relative 95% margin
    n_req = int(np.ceil(1.96**2 * (1 - p) / (p * cfg.target_rel_margin**2)))
    status = "PASS" if rel_m <= cfg.target_rel_margin else "FAIL"
    reason = (f"CTR={p:.4f} known to +-{rel_m*100:.1f}% (relative). "
              + ("Good enough." if status == "PASS" else
                 f"Target +-{cfg.target_rel_margin*100:.0f}% needs ~{n_req:,} rows."))
    return _res("precision", status, f"+-{rel_m*100:.1f}%",
                f"+-{cfg.target_rel_margin*100:.0f}%", reason,
                None if status == "PASS" else n_req)


def count_parameters(df, numeric_cols, categorical_cols):
    return len(numeric_cols) + sum(df[c].nunique() - 1 for c in categorical_cols)


def check_epv(df, y, numeric_cols, categorical_cols, cfg):
    k = count_parameters(df, numeric_cols, categorical_cols)
    events = int(min(y.sum(), (1 - y).sum()))
    epv = events / k if k else np.inf
    p = y.mean()
    n_req = int(np.ceil(cfg.min_epv * k / p)) if p > 0 else None
    status = "PASS" if epv >= cfg.min_epv else ("WARN" if epv >= 10 else "FAIL")
    reason = (f"{events:,} clicks / {k} parameters = {epv:.1f} events per variable. "
              + ("Stable fit." if status == "PASS" else
                 f"Need >= {cfg.min_epv}: ~{n_req:,} rows at this CTR, "
                 f"or reduce parameters (bucket rare category levels)."))
    return _res("EPV", status, f"{epv:.1f}", f">={cfg.min_epv}", reason,
                None if status == "PASS" else n_req)


def check_levels(df, y, categorical_cols, cfg):
    n, p = len(df), y.mean()
    worst, msgs, n_req = "PASS", [], None
    for c in categorical_cols:
        clicks = df.groupby(c, observed=True)[y.name].sum()
        counts = df[c].value_counts()
        thin = clicks[clicks < cfg.min_clicks_per_level]
        singletons = int((counts == 1).sum())
        unseen = singletons / n                       # Good-Turing estimate
        if len(thin) or unseen > cfg.unseen_mass_max:
            worst = "FAIL"
            s_min = cfg.min_clicks_per_level / (n * p) if p > 0 else np.nan
            if not np.isfinite(s_min) or s_min >= 0.5:
                fix = (f"At this CTR even a 100%-share level cannot reach "
                       f"{cfg.min_clicks_per_level} clicks -- the binding problem "
                       f"is total clicks (see precision/EPV), not bucketing.")
            else:
                smallest_share = counts.min() / n
                n_req_c = int(np.ceil(
                    cfg.min_clicks_per_level / (smallest_share * p)))
                fix = (f"Fix: bucket levels with share <{s_min*100:.2f}% into "
                       f"OTHER (cheap), or collect ~{n_req_c:,} rows to keep the "
                       f"rarest level separate (usually not worth it).")
            msgs.append(
                f"'{c}': {len(thin)}/{len(clicks)} levels have <"
                f"{cfg.min_clicks_per_level} clicks; unseen-level risk "
                f"(Good-Turing) = {unseen:.4f}. {fix}")
        else:
            msgs.append(f"'{c}': all {len(clicks)} levels have >="
                        f"{cfg.min_clicks_per_level} clicks; unseen risk {unseen:.4f}.")
    return _res("levels", worst,
                "; ".join(f"{c}:{df[c].nunique()} lvls" for c in categorical_cols),
                f">={cfg.min_clicks_per_level} clicks/level", " | ".join(msgs), n_req)


def check_ranges(df, numeric_cols, pred_ranges, cfg):
    pred_ranges = pred_ranges or {}
    worst, msgs = "PASS", []
    for c in numeric_cols:
        lo, hi = pred_ranges.get(
            c, (df[c].quantile(0.005), df[c].quantile(0.995)))
        edges = np.linspace(lo, hi, 11)               # 10 bins of the PREDICTION range
        counts, _ = np.histogram(df[c], bins=edges)
        bad = counts < cfg.min_rows_per_range_bin
        if bad.any():
            worst = "FAIL"
            gaps = [f"[{edges[i]:.1f},{edges[i+1]:.1f})"
                    for i in range(10) if bad[i]]
            msgs.append(
                f"'{c}': {bad.sum()}/10 bins of the prediction range "
                f"[{lo:.1f},{hi:.1f}] have <{cfg.min_rows_per_range_bin} training "
                f"rows: {', '.join(gaps[:4])}{'...' if len(gaps) > 4 else ''}. "
                f"The model would extrapolate there. Fix: collect data in the "
                f"gaps or restrict predictions to the covered range.")
    if not msgs:
        msgs.append("all numeric prediction ranges supported by training rows.")
    return _res("ranges", worst, f"{len(numeric_cols)} features checked",
                f">={cfg.min_rows_per_range_bin} rows/bin", " | ".join(msgs), None)


def check_power(y, cfg):
    n, p = len(y), y.mean()
    delta = cfg.target_delta_rel * p                  # absolute CTR diff of interest
    n_group_req = int(np.ceil(16 * p * (1 - p) / delta**2)) if delta > 0 else None
    n_small = int(n * min(cfg.power_group_share, 1 - cfg.power_group_share))
    mde = np.sqrt(16 * p * (1 - p) / n_small) if n_small else np.inf
    status = "PASS" if n_small >= (n_group_req or np.inf) else "FAIL"
    n_req = None if status == "PASS" else int(np.ceil(
        n_group_req / min(cfg.power_group_share, 1 - cfg.power_group_share)))
    reason = (f"Smallest detectable CTR difference with current groups: "
              f"{mde:.4f} (abs). You asked to detect {delta:.4f}. "
              + ("OK." if status == "PASS" else
                 f"Need ~{n_group_req:,} rows per group => ~{n_req:,} rows total."))
    return _res("power", status, f"MDE={mde:.4f}", f"delta={delta:.4f}",
                reason, n_req)


def check_learning_curve(df, y, numeric_cols, categorical_cols, cfg, seed=0):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb

    X = pd.get_dummies(df[numeric_cols + categorical_cols],
                       columns=categorical_cols, drop_first=True).astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)
    pts = []
    for f in cfg.lc_fractions:
        m = max(int(len(X_tr) * f), 50)
        mdl = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.07,
                                 class_weight="balanced",
                                 random_state=seed, verbose=-1)
        mdl.fit(X_tr.iloc[:m], y_tr.iloc[:m])
        pts.append((m, roc_auc_score(y_te, mdl.predict_proba(X_te)[:, 1])))
    ns = np.array([a for a, _ in pts]); aucs = np.array([b for _, b in pts])
    rising = (aucs[-1] - aucs[-2] > cfg.lc_rise_eps) or \
             (aucs[-1] - aucs[-3] > 2 * cfg.lc_rise_eps)
    curve = ", ".join(f"{a}:{b:.3f}" for a, b in pts)
    note = (" Curve is noisy (few clicks per point); treat numbers as rough."
            if np.any(np.diff(aucs) < -0.02) else "")

    if aucs[-1] >= cfg.target_auc:                    # target already met
        extra = "still rising -- more rows are optional bonus" if rising else "flat"
        return _res("learning_curve", "PASS", f"AUC {aucs[-1]:.3f}",
                    f">={cfg.target_auc}",
                    f"Target reached: AUC {aucs[-1]:.3f} ({extra}) "
                    f"({curve}).{note}", None)

    if rising:                                        # below target, still climbing
        n_req, tail = None, ""
        try:                                          # AUC(n) = A - B * n^(-beta)
            from scipy.optimize import curve_fit
            fun = lambda n, A, B, b: A - B * n**(-b)
            (A, B, b), _ = curve_fit(
                fun, ns, aucs, p0=[aucs[-1] + 0.03, 1.0, 0.5],
                bounds=([aucs[-1], 1e-6, 0.05], [1.0, 1e3, 2.0]), maxfev=20000)
            if A > cfg.target_auc:
                n_fit = int((B / (A - cfg.target_auc))**(1 / b))
                if n_fit > 20 * ns[-1]:               # too far to trust the fit
                    tail = (f" Power-law fit says ~{n_fit:,} rows -- far beyond "
                            f"reliable extrapolation; read as 'a lot more', "
                            f"collect in steps and re-check.")
                else:
                    n_req = n_fit
                    tail = (f" Power-law fit: plateau ~{A:.3f}; reaching AUC "
                            f"{cfg.target_auc} needs ~{n_req:,} training rows.")
            else:
                tail = (f" Power-law fit: plateau ~{A:.3f} < target "
                        f"{cfg.target_auc} => rows alone will not reach it; "
                        f"new features needed too.")
        except Exception:
            tail = " (extrapolation fit failed; collect more and re-check)"
        return _res("learning_curve", "WARN", f"AUC {aucs[-1]:.3f}, rising",
                    f">={cfg.target_auc}",
                    f"Below target and still rising ({curve}). More rows WILL "
                    f"help.{tail}{note}", n_req)

    return _res("learning_curve", "FAIL", f"AUC {aucs[-1]:.3f}, flat",
                f">={cfg.target_auc}",
                f"Curve is FLAT at {aucs[-1]:.3f} < target {cfg.target_auc} "
                f"({curve}). More rows will NOT help: the missing thing is "
                f"FEATURES (new signals), not rows.{note}", None)


# ----------------------------------------------------------------- audit
def audit(df, click_col, numeric_cols, categorical_cols,
          pred_ranges=None, cfg=AuditConfig(), run_learning_curve=True):
    y = df[click_col]
    checks = [
        check_precision(y, cfg),
        check_epv(df, y, numeric_cols, categorical_cols, cfg),
        check_levels(df, y, categorical_cols, cfg),
        check_ranges(df, numeric_cols, pred_ranges, cfg),
        check_power(y, cfg),
    ]
    if run_learning_curve:
        checks.append(check_learning_curve(df, y, numeric_cols,
                                           categorical_cols, cfg))
    rep = pd.DataFrame(checks)

    n_reqs = [c["n_req"] for c in checks if c["n_req"]]
    fails = rep[rep.status == "FAIL"]["check"].tolist()
    warns = rep[rep.status == "WARN"]["check"].tolist()
    lc = next((c for c in checks if c["check"] == "learning_curve"), None)
    feature_gap = lc and lc["status"] == "FAIL"

    if fails:
        verdict = f"FAIL -- failed: {', '.join(fails)}"
        if warns:
            verdict += f"; warnings: {', '.join(warns)}"
        verdict += "."
        if feature_gap:
            verdict += (" Main gap is INFORMATION, not size: add features; "
                        "more rows will not reach the target AUC.")
        elif n_reqs and max(n_reqs) > len(df):
            verdict += (f" Estimated required size: ~{max(n_reqs):,} rows "
                        f"(have {len(df):,}).")
    elif warns:
        verdict = (f"PASS WITH WARNINGS ({', '.join(warns)}) -- core "
                   f"requirements met with {len(df):,} rows / "
                   f"{int(y.sum()):,} clicks; see reasons above.")
    else:
        verdict = (f"PASS -- data is sufficient: {len(df):,} rows / "
                   f"{int(y.sum()):,} clicks cover precision, parameters, "
                   f"levels, ranges and power.")
    return rep, (max(n_reqs) if n_reqs else None), verdict


def print_audit(title, df, click_col, num, cat, pred_ranges=None,
                cfg=AuditConfig(), run_lc=True):
    print("\n" + "=" * 78 + f"\nSCENARIO: {title}\n" + "=" * 78)
    p = df[click_col].mean()
    print(f"rows={len(df):,}  clicks={int(df[click_col].sum()):,}  CTR={p:.4f}")
    rep, n_req, verdict = audit(df, click_col, num, cat, pred_ranges, cfg, run_lc)
    for _, r in rep.iterrows():
        print(f"[{r['status']:4}] {r['check']:<15} measured={r['measured']}  "
              f"required={r['required']}")
        print(f"       {r['reason']}")
    print(f"\nVERDICT: {verdict}")


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    rng = np.random.default_rng(11)

    def clicks_from(logit, r):
        return r.binomial(1, 1 / (1 + np.exp(-logit)))

    # A) healthy ---------------------------------------------------------
    n = 60_000; r = np.random.default_rng(1)
    price = r.gamma(4, 25, n); rating = np.clip(r.normal(4, .6, n), 1, 5)
    novelty = r.uniform(0, 10, n)
    brand = r.choice([f"b{i}" for i in range(8)], n,
                     p=[.25, .2, .15, .12, .1, .08, .06, .04])
    device = r.choice(["mobile", "desktop", "tablet"], n, p=[.6, .3, .1])
    beff = dict(zip([f"b{i}" for i in range(8)], r.normal(0, .25, 8)))
    lg = (-2.2 - 0.01*(price-100) + 0.8*(rating-4) + 0.4 - 0.05*(novelty-5)**2
          + pd.Series(brand).map(beff).to_numpy())
    dfA = pd.DataFrame({"f_price": price, "f_rating": rating, "f_novelty": novelty,
                        "brand": brand, "device": device,
                        "click": clicks_from(lg, r)})
    print_audit("A. healthy dataset (should PASS everything)",
                dfA, "click", ["f_price", "f_rating", "f_novelty"],
                ["brand", "device"])

    # B) too few clicks --------------------------------------------------
    n = 4_000; r = np.random.default_rng(2)
    price = r.gamma(4, 25, n); rating = np.clip(r.normal(4, .6, n), 1, 5)
    quality = r.normal(0, 1, n)
    brand = r.choice([f"b{i}" for i in range(5)], n)
    device = r.choice(["mobile", "desktop", "tablet"], n, p=[.6, .3, .1])
    lg = -4.9 - 0.006*(price-100) + 0.5*(rating-4) + 0.2*quality
    dfB = pd.DataFrame({"f_price": price, "f_rating": rating, "f_quality": quality,
                        "brand": brand, "device": device,
                        "click": clicks_from(lg, r)})
    print_audit("B. rare clicks, small n (should FAIL precision/EPV/power)",
                dfB, "click", ["f_price", "f_rating", "f_quality"],
                ["brand", "device"])

    # C) long-tail categories --------------------------------------------
    n = 30_000; r = np.random.default_rng(3)
    n_brands = 300
    shares = 1 / np.arange(1, n_brands + 1)**1.15; shares /= shares.sum()
    brand = r.choice([f"b{i}" for i in range(n_brands)], n, p=shares)
    price = r.gamma(4, 25, n); rating = np.clip(r.normal(4, .6, n), 1, 5)
    beff = dict(zip([f"b{i}" for i in range(n_brands)],
                    r.normal(0, .3, n_brands)))
    lg = (-2.9 - 0.008*(price-100) + 0.6*(rating-4)
          + pd.Series(brand).map(beff).to_numpy())
    dfC = pd.DataFrame({"f_price": price, "f_rating": rating, "brand": brand,
                        "click": clicks_from(lg, r)})
    print_audit("C. 300 long-tail brand levels (should FAIL levels & EPV)",
                dfC, "click", ["f_price", "f_rating"], ["brand"])

    # D) range gap --------------------------------------------------------
    n = 25_000; r = np.random.default_rng(4)
    price = r.uniform(10, 60, n)                       # training only covers 10..60
    novelty = np.concatenate([r.uniform(0, 3, n//2),   # internal hole 3..7
                              r.uniform(7, 10, n - n//2)])
    rating = np.clip(r.normal(4, .6, n), 1, 5)
    device = r.choice(["mobile", "desktop"], n, p=[.6, .4])
    lg = -2.3 - 0.02*(price-35) + 0.7*(rating-4)
    dfD = pd.DataFrame({"f_price": price, "f_novelty": novelty,
                        "f_rating": rating, "device": device,
                        "click": clicks_from(lg, r)})
    print_audit("D. prediction range wider than training (should FAIL ranges)",
                dfD, "click", ["f_price", "f_novelty", "f_rating"], ["device"],
                pred_ranges={"f_price": (10, 200), "f_novelty": (0, 10)})

    # E) still rising ------------------------------------------------------
    n = 2_500; r = np.random.default_rng(5)
    price = r.gamma(4, 25, n); rating = np.clip(r.normal(4, .6, n), 1, 5)
    novelty = r.uniform(0, 10, n)
    delivery = r.integers(1, 10, n).astype(float)
    device = r.choice(["mobile", "desktop", "tablet"], n, p=[.6, .3, .1])
    lg = (-2.1 - 0.012*(price-100) + 0.9*(rating-4) + 0.5 - 0.055*(novelty-5)**2
          - 0.16*(delivery-5)*(device == "mobile"))
    dfE = pd.DataFrame({"f_price": price, "f_rating": rating, "f_novelty": novelty,
                        "f_delivery_days": delivery, "device": device,
                        "click": clicks_from(lg, r)})
    print_audit("E. rich signal but only 2,500 rows (learning curve should RISE)",
                dfE, "click",
                ["f_price", "f_rating", "f_novelty", "f_delivery_days"],
                ["device"])

    # F) weak features ------------------------------------------------------
    n = 40_000; r = np.random.default_rng(6)
    x1, x2, x3 = r.normal(0, 1, n), r.normal(0, 1, n), r.normal(0, 1, n)
    device = r.choice(["mobile", "desktop"], n, p=[.5, .5])
    lg = -2.45 + 0.06*x1 + 0.05*x2 + 0.04*x3
    dfF = pd.DataFrame({"f_x1": x1, "f_x2": x2, "f_x3": x3, "device": device,
                        "click": clicks_from(lg, r)})
    print_audit("F. 40k rows but nearly useless features "
                "(curve should be FLAT BELOW target => missing features)",
                dfF, "click", ["f_x1", "f_x2", "f_x3"], ["device"])
