import numpy as np
from typing import Literal

def pairwise_euclidean_distance_with_mask(
    X: np.ndarray
) -> np.ndarray:
    """
    X: (n_samples, n_features)  Allows arbitrary NaN
    Returns: (n_samples, n_samples)  Euclidean distance matrix
    """
    # 1) Construct global mask: 1=present 0=missing
    W = (~np.isnan(X)).astype(bool)

    # 2) Get number of samples
    n = X.shape[0]

    # 3) Initialize distance matrix
    dist = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(dist, 0.0)

    # 4) Compute pairwise Euclidean distance
    for i in range(n):
        for j in range(i + 1, n):
            # Common features present in both samples
            common_mask = W[i] & W[j]
            if np.sum(common_mask) == 0:
                # No common features → set distance to infinity (completely dissimilar)
                dist[i, j] = dist[j, i] = np.inf
            else:
                # Extract common features
                x1_common = X[i][common_mask]
                x2_common = X[j][common_mask]
                # Compute Euclidean distance
                d = np.linalg.norm(x1_common - x2_common)
                dist[i, j] = dist[j, i] = d

    return dist