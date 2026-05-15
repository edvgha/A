import numpy as np
from scipy.stats import multivariate_normal

class SoftKMeans:
    def __init__(self, k, max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, data):
        n_samples, n_features = data.shape
        
        # Initialization
        # Means: Random points from data
        self.means = data[np.random.choice(n_samples, self.k, replace=False)]
        # Covariances: Initialize as Identity matrices (spheres)
        self.covs = np.array([np.eye(n_features) for _ in range(self.k)])
        # Mixing coefficients: Equal weights
        self.pi = np.ones(self.k) / self.k
        
        for iteration in range(self.max_iter):
            prev_means = self.means.copy()

            # --- E-STEP: Responsibility r_k^n ---
            resp = np.zeros((n_samples, self.k))
            for k in range(self.k):
                # Using scipy for robust multivariate PDF calculation
                resp[:, k] = self.pi[k] * multivariate_normal.pdf(data, self.means[k], self.covs[k])
            
            # Normalize responsibilities
            resp /= (np.sum(resp, axis=1, keepdims=True) + 1e-10)

            # --- M-STEP: Parameter Updates ---
            R_k = np.sum(resp, axis=0) # Total weight for each cluster

            for k in range(self.k):
                # Update Mean: m = sum(r * x) / R
                self.means[k] = np.sum(resp[:, [k]] * data, axis=0) / R_k[k]
                
                # Update Full Covariance Matrix: Sigma = sum(r * (x-m)(x-m).T) / R
                diff = data - self.means[k]
                # Vectorized outer product update
                self.covs[k] = (resp[:, k] * diff.T) @ diff / R_k[k]
                
                # Add small epsilon to diagonal for numerical stability (ensure positive-definite)
                self.covs[k] += np.eye(n_features) * 1e-6

            # Update Pi: pi = R / N
            self.pi = R_k / n_samples

            if np.linalg.norm(self.means - prev_means) < self.tol:
                print(f'Converged at iteration {iteration}')
                break
        return resp # type: ignore
