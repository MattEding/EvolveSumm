import numba
import numpy as np


@numba.njit
def jaccard_similarity(a, b):
    union = np.sum(a | b)
    if not union:
        return 1.0
    return np.sum(a & b) / union


@numba.njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@numba.njit
def cohesion(chromosome, similarity, document):
    total = 0
    for p in np.unique(chromosome):
        sents = document[chromosome == p]
        k = len(sents)
        #: combinations(sents, r=2)
        for i in range(k-1):
            for j in range(i+1, k):
                total += similarity(sents[i], sents[j]) / len(sents)
    return total


@numba.njit
def separation(chromosome, similarity, document):
    total = 0
    k = len(np.unique(chromosome))
    #: combinations(k, r=2)
    for p in range(k-1):
        for q in range(p+1, k):
            sents_p = document[chromosome == p]
            sents_q = document[chromosome == q]
            #: product(sents_p, sents_q)
            m, n = len(sents_p), len(sents_q)
            for i in range(m):
                for j in range(n):
                    total += similarity(sents_p[i], sents_q[j]) / m / n
    return total


@numba.njit
def cohesion_separation(chromosome, similarity, document):
    coh = cohesion(chromosome, similarity, document)
    sep = separation(chromosome, similarity, document)
    return (1 + sigmoid(coh)) ** sep
