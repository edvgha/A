"""Multinomial Naive Bayes with Laplace (add-alpha) smoothing.

Training is a single counting pass; prediction is a matrix multiply in log
space.
"""
import jax
import jax.numpy as jnp
from functools import partial


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """X: (n, V) count matrix.  y: (n,) integer class labels in [0, C)."""
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y)
        C = int(y.max()) + 1
        V = X.shape[1]

        # One-hot the labels -> (n, C), then sum to get per-class aggregates.
        Y = jax.nn.one_hot(y, C)                       # (n, C)

        # Class priors: log P(c) = log(N_c / N).
        class_counts = Y.sum(axis=0)                   # (C,)
        self.log_prior = jnp.log(class_counts / class_counts.sum())  # (C,)

        # Feature counts per class: T[c, i] = sum of feature i over class c.
        # (C, n) @ (n, V) -> (C, V).
        T = Y.T @ X                                    # (C, V)

        # Smoothed likelihood:  (T_ci + alpha) / (sum_j T_cj + alpha*V).
        num = T + self.alpha                           # (C, V)
        den = num.sum(axis=1, keepdims=True)           # (C, 1) == row sum
        self.log_likelihood = jnp.log(num / den)       # (C, V)
        return self

    @partial(jax.jit, static_argnums=0)
    def _log_posterior(self, X):
        # log P(c|x) ∝ log P(c) + sum_i x_i * log P(i|c)
        # (n, V) @ (V, C) -> (n, C), then add the prior row.
        return X @ self.log_likelihood.T + self.log_prior   # (n, C)

    def predict_log_proba(self, X):
        return self._log_posterior(jnp.asarray(X, dtype=jnp.float32))

    def predict(self, X):
        return jnp.argmax(self.predict_log_proba(X), axis=1)


if __name__ == "__main__":
    # Tiny text example: 3 docs/class, vocab of 5 words.
    # Classes: 0 = "sports", 1 = "tech".
    vocab = ["game", "score", "win", "cpu", "code"]
    X_train = jnp.array([
        [3, 2, 2, 0, 0],   # sports
        [2, 3, 1, 0, 0],   # sports
        [4, 1, 2, 0, 0],   # sports
        [0, 0, 0, 3, 4],   # tech
        [0, 0, 1, 2, 3],   # tech
        [0, 0, 0, 4, 2],   # tech
    ], dtype=jnp.float32)
    y_train = jnp.array([0, 0, 0, 1, 1, 1])

    # Test doc contains "win" + "cpu"+"code". Note "win" never co-occurs with
    # tech words in training, and tech never saw "game/score".
    X_test = jnp.array([[0, 0, 1, 2, 2]], dtype=jnp.float32)

    print("=== with smoothing (alpha=1.0) ===")
    m = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    lp = m.predict_log_proba(X_test)
    print("log-posteriors [sports, tech]:", lp)
    print("prediction:", ["sports", "tech"][int(m.predict(X_test)[0])])

    print("\n=== zero-frequency problem (alpha=0) ===")
    # NB: with alpha=0, unseen (class,word) pairs give log P = log(0) = -inf.
    # A word with COUNT>0 in the test doc but zero prob in a class -> -inf
    # cleanly vetoes that class. But a word with count 0 gives 0*(-inf) = nan,
    # which poisons argmax. So unsmoothed multinomial NB is not just risky,
    # it's numerically broken — this is why alpha>0 is effectively mandatory.
    m0 = MultinomialNB(alpha=0.0).fit(X_train, y_train)
    print("log-likelihood matrix (rows=class, cols=word):")
    print(m0.log_likelihood)
    lp0 = m0.predict_log_proba(X_test)
    print("log-posteriors [sports, tech]:", lp0,
          "  <- -inf/nan from zero counts: the model is unusable")
