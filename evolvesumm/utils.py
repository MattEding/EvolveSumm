import numba as nb
import numpy as np


@nb.njit(nb.float32(nb.boolean[::1], nb.boolean[::1]), fastmath=True)
def _jaccard_similarity(a, b):
    """Return the ratio of the intersection to the union of two containers.
    | a ∩ b |
    —————————
    | a ∪ b |
    """

    union = np.sum(a | b)
    if not union:
        result = 1.0
    else:
        result = np.sum(a & b) / union
    return result


@nb.guvectorize(
    ['void(float64[::1], boolean[:, ::1], float32[::1])'], 
    '(n),(n,m)->()', 
    target='parallel', 
    fastmath=True)
def cohesion(chrom, doc, total):
    """Measure of how compact all the clusters are.
     k            sim(Si, Sj)
     ∑      ∑     ———————————
    p=1  Si,Sj∈Cp    |Cp|
    """

    total[0] = 0.0
    for p in np.unique(chrom):
        sents = doc[chrom == p]
        k = len(sents)
        #: itertools.combinations(sents, r=2) for numba
        for i in range(k-1):
            for j in range(i+1, k):
                total[0] += _jaccard_similarity(sents[i], sents[j]) / k


@nb.guvectorize(
    ['void(float64[::1], boolean[:, ::1], float32[::1])'], 
    '(n),(n,m)->()', 
    target='parallel', 
    fastmath=True)
def separation(chrom, doc, total):
    """Measure of how separable all the clusters are.
    k-1     k                 sim(Si, Sj)
     ∑      ∑      ∑      ∑   ———————————
    p=1   q=p+1  Si∈Cp  Sj∈Cq  |Cp|·|Cp|
    """

    total[0] = 0.0
    k = len(np.unique(chrom))
    #: itertools.combinations(doc, r=2) for numba
    for p in range(k-1):
        for q in range(p+1, k):
            sents_p = doc[chrom == p]
            sents_q = doc[chrom == q]
            #: itertools.product(sents_p, sents_q) for numba
            m, n = len(sents_p), len(sents_q)
            for i in range(m):
                for j in range(n):
                    total[0] += _jaccard_similarity(sents_p[i], sents_q[j]) / m / n
                    

def sigmoid(x):
    """Sigmoid function defined as 1 / (1 + exp(-x))."""

    return 1 / (1 + np.exp(-x))


def cohesion_separation(chroms, doc):
    """Measure balancing both cohesion and separation of clusters."""

    coh = cohesion(chroms, doc)
    sep = separation(chroms, doc)
    return (1 + sigmoid(coh)) ** sep
