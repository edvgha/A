"""Bernoulli Naive Bayes with add-alpha (Beta-prior) smoothing.

Features are binary: x_i in {0, 1}

Score:
    log P(c|x) = log P(c) + sum_i [ x_i*log(phi_ci) + (1-x_i)*log(1-phi_ci) ]
"""
import jax
import jax.numpy as jnp
from functools import partial


class BernoulliNB:
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha          # smoothing strength (>0 strongly advised)
        self.binarize = binarize    # threshold: x > binarize -> 1, else 0.
                                    # set to None if input is already binary.

    def _binarize(self, X):
        if self.binarize is None:
            return jnp.asarray(X, dtype=jnp.float32)
        return (jnp.asarray(X, dtype=jnp.float32) > self.binarize).astype(jnp.float32)

    def fit(self, X, y):
        """X: (n, V) feature matrix.  y: (n,) integer class labels in [0, C)."""
        X = self._binarize(X)
        y = jnp.asarray(y)
        C = int(y.max()) + 1

        # One-hot labels -> (n, C); column sums give per-class aggregates.
        Y = jax.nn.one_hot(y, C)                       # (n, C)

        # Class priors: log P(c) = log(N_c / N).
        N_c = Y.sum(axis=0)                            # (C,)  samples per class
        self.log_prior = jnp.log(N_c / N_c.sum())      # (C,)

        # Per-class present-counts: N_ci = # class-c samples with feature i = 1.
        # (C, n) @ (n, V) -> (C, V).
        N_ci = Y.T @ X                                 # (C, V)

        # Smoothed P(x_i = 1 | c):  (N_ci + alpha) / (N_c + 2*alpha).
        phi = (N_ci + self.alpha) / (N_c[:, None] + 2 * self.alpha)  # (C, V)

        # Precompute both log terms used at scoring time.
        self.log_phi = jnp.log(phi)                    # (C, V)  present term
        self.log_1mphi = jnp.log(1.0 - phi)            # (C, V)  absent term
        return self

    @partial(jax.jit, static_argnums=0)
    def _log_posterior(self, X):
        # log P(c|x) = log P(c) + sum_i x_i*log(phi) + (1-x_i)*log(1-phi)
        # Rewrite the sum as two matmuls:
        #   X @ log_phi.T            -> contribution from PRESENT features
        #   (1-X) @ log_1mphi.T      -> contribution from ABSENT features
        present = X @ self.log_phi.T                   # (n, C)
        absent = (1.0 - X) @ self.log_1mphi.T          # (n, C)
        return present + absent + self.log_prior       # (n, C)

    def predict_log_proba(self, X):
        return self._log_posterior(self._binarize(X))

    def predict(self, X):
        return jnp.argmax(self.predict_log_proba(X), axis=1)


if __name__ == "__main__":
    # Binary presence/absence over a 5-word vocab. Classes: 0=sports, 1=tech.
    vocab = ["game", "score", "win", "cpu", "code"]
    X_train = jnp.array([
        [1, 1, 1, 0, 0],   # sports
        [1, 1, 0, 0, 0],   # sports
        [1, 0, 1, 0, 0],   # sports
        [0, 0, 0, 1, 1],   # tech
        [0, 0, 1, 1, 1],   # tech
        [0, 0, 0, 1, 1],   # tech
    ], dtype=jnp.float32)
    y_train = jnp.array([0, 0, 0, 1, 1, 1])

    m = BernoulliNB(alpha=1.0, binarize=None).fit(X_train, y_train) # type: ignore

    X_test = jnp.array([
        [1, 1, 0, 0, 0],   # game+score present -> sports
        [0, 0, 0, 1, 1],   # cpu+code present   -> tech
        [0, 0, 1, 0, 0],   # only 'win' (appears in both classes)
    ], dtype=jnp.float32)

    lp = m.predict_log_proba(X_test)
    preds = m.predict(X_test)
    names = ["sports", "tech"]
    for i, row in enumerate(X_test):
        print(f"doc {list(map(int, row))}: "
              f"log-post [sports, tech] = {jnp.round(lp[i], 3)} "
              f"-> {names[int(preds[i])]}")

    print("\nP(word=1 | class) matrix (rows=class, cols=word):")
    print(jnp.round(jnp.exp(m.log_phi), 3))
    print("vocab:", vocab)
