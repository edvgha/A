"""
Synthetic click data + the "alternating group-bys" estimator for
    P(click | offer o, position p) = theta_p * gamma_o
Demonstrates: (1) naive group-by CTR is confounded by the ranker,
(2) the alternating fixed point recovers the truth,
(3) it is identical to the Poisson GLM MLE,
(4) bootstrap uncertainty.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- ground truth
N = 300_000
issuers     = np.array(['x', 'y', 'z', 'e'])          # one offer per issuer
gamma_true  = np.array([0.090, 0.055, 0.030, 0.015])  # attractiveness = CTR at pos 1
theta_true  = np.array([1.00, 0.55, 0.30])            # examination, normalized th_1 = 1
share       = np.array([0.30, 0.30, 0.25, 0.15])      # traffic share per issuer

# Ranker-induced confounding: better issuers are shown higher (but every
# issuer appears at every position -> the offer-position graph is connected)
P_pos = np.array([[0.70, 0.20, 0.10],   # x mostly slot 1
                  [0.20, 0.50, 0.30],   # y mostly slot 2
                  [0.08, 0.25, 0.67],   # z mostly slot 3
                  [0.05, 0.15, 0.80]])  # e mostly slot 3

# ---------------------------------------------------------------- simulate rows
iss_idx = rng.choice(4, size=N, p=share)
u       = rng.random(N)
pos     = (u[:, None] > P_pos.cumsum(axis=1)[iss_idx]).sum(axis=1)   # 0,1,2
p_click = theta_true[pos] * gamma_true[iss_idx]
click   = (rng.random(N) < p_click).astype(int)

df = pd.DataFrame({'rank_pos': pos + 1, 'click': click, 'issuer': issuers[iss_idx]})
df.to_csv('/mnt/user-data/outputs/synthetic_clicks.csv', index=False)
print(f"rows: {len(df)}, overall CTR: {df.click.mean():.4f}")

# ---------------------------------------------------------------- naive group-bys
naive_pos = df.groupby('rank_pos')['click'].mean()
naive_iss = df.groupby('issuer')['click'].mean().reindex(issuers)
print("\nNAIVE CTR by position:\n", naive_pos.round(4).to_string())
print("naive ratios th_p/th_1:", (naive_pos / naive_pos.loc[1]).round(3).values,
      " <- truth:", theta_true)
print("\nNAIVE CTR by issuer:\n", naive_iss.round(4).to_string())
print("naive ratios g_o/g_x:", (naive_iss / naive_iss.loc['x']).round(3).values,
      " <- truth:", (gamma_true / gamma_true[0]).round(3))

# ---------------------------------------------------------------- sufficient stats
cells = (df.groupby(['issuer', 'rank_pos'])
           .agg(clicks=('click', 'sum'), imps=('click', 'size'))
           .reset_index())
C  = cells.pivot(index='issuer', columns='rank_pos', values='clicks') \
          .loc[issuers].values.astype(float)
Nm = cells.pivot(index='issuer', columns='rank_pos', values='imps') \
          .loc[issuers].values.astype(float)
print("\nimpressions N_op:\n", Nm.astype(int))
print("clicks C_op:\n", C.astype(int))

# ---------------------------------------------------------------- the algorithm
def fit_alternating(C, Nm, tol=1e-12, max_iter=1000, trace=False):
    theta = np.ones(C.shape[1])
    hist  = []
    for it in range(max_iter):
        gamma     = C.sum(axis=1) / (Nm @ theta)        # offer block, closed form
        theta_new = C.sum(axis=0) / (Nm.T @ gamma)      # position block, closed form
        hist.append(theta_new / theta_new[0])
        if np.max(np.abs(theta_new - theta)) < tol:
            theta = theta_new
            break
        theta = theta_new
    gamma = C.sum(axis=1) / (Nm @ theta)
    return theta / theta[0], gamma * theta[0], hist     # normalize th_1 = 1

theta_hat, gamma_hat, hist = fit_alternating(C, Nm, trace=True)

print("\nITERATION TRACE (normalized theta):")
for it in [0, 1, 2, 3, 4, 9, len(hist) - 1]:
    if it < len(hist):
        print(f"  iter {it+1:3d}: theta = {np.round(hist[it], 5)}")
print(f"converged in {len(hist)} iterations")

print("\nALTERNATING estimates:")
print("theta_hat:", theta_hat.round(4), " truth:", theta_true)
print("gamma_hat:", gamma_hat.round(4), " truth:", gamma_true)

# fixed-point / moment-matching check: fitted totals == observed totals
fitted = Nm * gamma_hat[:, None] * theta_hat[None, :]
print("\nmoment check  max|fitted-observed| row totals:",
      f"{np.max(np.abs(fitted.sum(1) - C.sum(1))):.2e}",
      " col totals:", f"{np.max(np.abs(fitted.sum(0) - C.sum(0))):.2e}")

# ---------------------------------------------------------------- Poisson GLM
import statsmodels.api as sm
import statsmodels.formula.api as smf
cells['issuer'] = pd.Categorical(cells['issuer'], categories=list(issuers))
cells['pos_c']  = pd.Categorical(cells['rank_pos'])
glm = smf.glm('clicks ~ pos_c + issuer', data=cells,
              offset=np.log(cells['imps']),
              family=sm.families.Poisson()).fit()
th_glm = np.exp(np.r_[0, glm.params['pos_c[T.2]'], glm.params['pos_c[T.3]']])
ga_glm = np.exp(glm.params['Intercept']
                + np.r_[0, glm.params['issuer[T.y]'],
                        glm.params['issuer[T.z]'], glm.params['issuer[T.e]']])
print("\nPOISSON GLM (same model, Newton-fitted):")
print("theta_glm:", th_glm.round(6))
print("theta_alt:", theta_hat.round(6), " max diff:",
      f"{np.max(np.abs(th_glm - theta_hat)):.2e}")
print("gamma_glm:", ga_glm.round(6), " max diff:",
      f"{np.max(np.abs(ga_glm - gamma_hat)):.2e}")

# ---------------------------------------------------------------- bootstrap CIs
# rows are iid here, so bootstrapping rows == multinomial resampling of the
# 24 categories (12 cells x {click, no click}); with real widget data,
# resample impression ids instead.
p24  = np.concatenate([C.ravel(), (Nm - C).ravel()]) / N
reps = 1000
boot = np.empty((reps, 5))                     # th2, th3, gy/gx, gz/gx, ge/gx
for b in range(reps):
    cnt = rng.multinomial(N, p24)
    Cb  = cnt[:12].reshape(4, 3).astype(float)
    Nb  = Cb + cnt[12:].reshape(4, 3)
    th_b, ga_b, _ = fit_alternating(Cb, Nb)
    boot[b] = [th_b[1], th_b[2], ga_b[1]/ga_b[0], ga_b[2]/ga_b[0], ga_b[3]/ga_b[0]]

lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
names  = ['theta_2/theta_1', 'theta_3/theta_1',
          'gamma_y/gamma_x', 'gamma_z/gamma_x', 'gamma_e/gamma_x']
truthv = [theta_true[1], theta_true[2],
          gamma_true[1]/gamma_true[0], gamma_true[2]/gamma_true[0],
          gamma_true[3]/gamma_true[0]]
est    = [theta_hat[1], theta_hat[2],
          gamma_hat[1]/gamma_hat[0], gamma_hat[2]/gamma_hat[0],
          gamma_hat[3]/gamma_hat[0]]
print("\nBOOTSTRAP 95% CIs (1000 reps):")
for n, e, l, h, t in zip(names, est, lo, hi, truthv):
    print(f"  {n}: {e:.3f}  [{l:.3f}, {h:.3f}]   truth {t:.3f}")
