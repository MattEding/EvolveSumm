import numpy as np
from numba import njit


@njit
def jaccard(a, b):
    """Return the ratio of the intersection to the union of two containers."""
    union = np.sum(a | b)
    if not union:
        return 1.0
    return np.sum(a & b) / union


@njit
def sigmoid(x):
    """Sigmoid function defined as 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-x))


@njit
def cohesion(chrom, doc, sim):
    """Measure of how compact all the clusters are."""
    total = 0
    for p in np.unique(chrom):
        sents = doc[chrom == p]
        k = len(sents)
        #: itertools.combinations(sents, r=2) for numba
        for i in range(k-1):
            for j in range(i+1, k):
                total += sim(sents[i], sents[j]) / len(sents)
    return total


@njit
def separation(chrom, doc, sim):
    """Measure of how separable all the clusters are."""
    total = 0
    k = len(np.unique(chrom))
    #: itertools.combinations(k, r=2) for numba
    for p in range(k-1):
        for q in range(p+1, k):
            sents_p = doc[chrom == p]
            sents_q = doc[chrom == q]
            #: itertools.product(sents_p, sents_q) for numba
            m, n = len(sents_p), len(sents_q)
            for i in range(m):
                for j in range(n):
                    total += sim(sents_p[i], sents_q[j]) / m / n
    return total


@njit
def cohesion_separation(chrom, doc, sim):
    """Measure balancing both cohesion and separation of clusters."""
    coh = cohesion(chrom, doc, sim)
    sep = separation(chrom, doc, sim)
    return (1 + sigmoid(coh)) ** sep
