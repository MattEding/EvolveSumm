from numba import njit


@njit
def jaccard(a, b):
    union = np.sum(a | b)
    if not union:
        return 1.0
    return np.sum(a & b) / union


@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
