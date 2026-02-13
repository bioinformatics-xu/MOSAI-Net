import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal

def pairwise_cosine_similarity_with_mask(
    X: np.ndarray
) -> np.ndarray:
    """
    X: (n_samples, n_features)  Allows arbitrary NaN
    Returns: (n_samples, n_samples)  Cosine similarity matrix
    """
    # 1) Construct global weight matrix: 1=present 0=missing
    W = (~np.isnan(X)).astype(bool)  # Ensure W is boolean type

    # 2) Get number of samples
    n = X.shape[0]

    # 3) Initialize similarity matrix
    sim = np.eye(n, dtype=float)

    # 4) Compute pairwise cosine similarity
    for i in range(n):
        for j in range(i + 1, n):
            # Select features common to both samples
            common_mask = W[i] & W[j]  # Use boolean logical AND operation
            if np.sum(common_mask) == 0:
                # If no common features, similarity is NaN
                sim[i, j] = sim[j, i] = np.nan
            else:
                # Extract common features
                x1_common = X[i][common_mask]
                x2_common = X[j][common_mask]
                # Compute standard cosine similarity
                similarity = cosine_similarity([x1_common], [x2_common])[0, 0]
                sim[i, j] = sim[j, i] = similarity

    return sim