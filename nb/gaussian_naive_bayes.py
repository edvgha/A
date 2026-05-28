"""Gaussian Naive Bayes.

Score:

    log P(c|x) = log P(c)
                 + sum_i [ -0.5*log(2*pi*var_ci) - (x_i - mu_ci)^2 / (2*var_ci) ]
"""
import jax
import jax.numpy as jnp
from functools import partial


class GaussianNB:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def fit(self, X, y):
        """X: (n, d) continuous features.  y: (n,) integer labels in [0, C)."""
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y)
        C = int(y.max()) + 1

        # One-hot labels -> (n, C); lets us compute per-class stats by matmul.
        Y = jax.nn.one_hot(y, C)                       # (n, C)
        N_c = Y.sum(axis=0)                            # (C,)  samples per class

        # Class priors: log P(c) = log(N_c / N).
        self.log_prior = jnp.log(N_c / N_c.sum())      # (C,)

        # Per-class mean:  mu_ci = (1/N_c) * sum_{s in c} x_i.
        # (C, n) @ (n, d) -> (C, d), divided row-wise by N_c.
        sums = Y.T @ X                                 # (C, d)  sum of x per class
        mu = sums / N_c[:, None]                       # (C, d)

        # Per-class variance:  var_ci = (1/N_c) * sum_{s in c} (x_i - mu_ci)^2.
        # E[x^2] - mu^2 is one matmul; equivalent to the sum-of-squares form.
        sq_sums = Y.T @ (X ** 2)                       # (C, d)  sum of x^2
        var = sq_sums / N_c[:, None] - mu ** 2         # (C, d)  E[x^2]-E[x]^2

        # Variance floor: add epsilon * (largest feature variance), so the
        # floor scales with the data and never lets var hit 0.
        floor = self.epsilon * X.var(axis=0).max()
        var = var + floor                              # (C, d)

        self.mu = mu                                   # (C, d)
        self.var = var                                 # (C, d)
        # Precompute the constant -0.5*log(2*pi*var) term used at scoring.
        self.log_norm = -0.5 * jnp.log(2.0 * jnp.pi * var)   # (C, d)
        return self

    @partial(jax.jit, static_argnums=0)
    def _log_posterior(self, X):
        # For each class c, sum the per-feature Gaussian log-densities:
        #   sum_i [ -0.5*log(2*pi*var_ci) - (x_i - mu_ci)^2 / (2*var_ci) ]
        # Broadcast X (n,1,d) against params (1,C,d) -> (n,C,d), sum over d.
        Xe = X[:, None, :]                             # (n, 1, d)
        mu = self.mu[None, :, :]                        # (1, C, d)
        var = self.var[None, :, :]                      # (1, C, d)
        logn = self.log_norm[None, :, :]                # (1, C, d)
        log_density = logn - (Xe - mu) ** 2 / (2.0 * var)   # (n, C, d)
        return log_density.sum(axis=2) + self.log_prior     # (n, C)

    def predict_log_proba(self, X):
        return self._log_posterior(jnp.asarray(X, dtype=jnp.float32))

    def predict(self, X):
        return jnp.argmax(self.predict_log_proba(X), axis=1)


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[0, 0], scale=[1.0, 1.5], size=(100, 2))
    X1 = rng.normal(loc=[5, 5], scale=[1.5, 1.0], size=(100, 2))
    X_train = jnp.asarray(np.vstack([X0, X1]), dtype=jnp.float32)
    y_train = jnp.asarray(np.array([0] * 100 + [1] * 100))

    m = GaussianNB().fit(X_train, y_train)

    print("fitted means (rows=class):")
    print(jnp.round(m.mu, 2))
    print("fitted variances (rows=class):")
    print(jnp.round(m.var, 2))

    X_test = jnp.array([[0.0, 0.0], [5.0, 5.0], [2.5, 2.5]], dtype=jnp.float32)
    lp = m.predict_log_proba(X_test)
    preds = m.predict(X_test)
    for i, q in enumerate(X_test):
        print(f"point {list(map(float, q))}: "
              f"log-post = {jnp.round(lp[i], 2)} -> class {int(preds[i])}")

    # Accuracy on the training set (sanity check).
    acc = float((m.predict(X_train) == y_train).mean())
    print(f"\ntrain accuracy: {acc:.3f}")
