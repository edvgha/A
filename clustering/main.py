import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from data import generate_cigar_data
from soft_k_means import SoftKMeans

def show(data: np.ndarray, model: SoftKMeans, resp: np.ndarray):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], c=resp[:, 0], cmap='viridis', alpha=0.4)

    for k in range(model.k):
        rv = multivariate_normal(model.means[k], model.covs[k])
        z = rv.pdf(grid).reshape(xx.shape)
        plt.contour(xx, yy, z, levels=5, colors='black', alpha=0.8)

    plt.scatter(model.means[:, 0], model.means[:, 1], marker='X', s=200, color='red', label='Centroids')
    plt.title("Soft K-Means with Full Covariance (Tilted Contours)")
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    data = generate_cigar_data()
    model = SoftKMeans(k=2)
    resp = model.fit(data)
    show(data, model, resp)
