import numpy as np

def generate_cigar_data(n_samples=300, seed=42):
    np.random.seed(seed)
    
    # Cluster 1: Elongated diagonally (y = x)
    # High covariance between x and y creates the "cigar" tilt
    mean1 = [3, 3]
    cov1 = [[5, 4.5], [4.5, 5]] 
    cluster1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    
    # Cluster 2: Elongated horizontally
    # Large variance in X, very small variance in Y
    mean2 = [-3, -3]
    cov2 = [[8, 0.5], [0.5, 0.5]]
    cluster2 = np.random.multivariate_normal(mean2, cov2, n_samples)
    
    # Stack together for a single dataset
    data = np.vstack([cluster1, cluster2])
    return data
