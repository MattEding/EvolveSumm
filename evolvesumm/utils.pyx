import numpy as np

from cython cimport boundscheck, wraparound
from cython.parallel cimport prange
cimport numpy as cnp

from libc.math cimport exp

cimport libc.math as cmath


@boundscheck(False)
@wraparound(False)
cdef double _jaccard_similarity(int[::1] a, int[::1] b) nogil:
    """Return the ratio of the intersection to the union of two containers.

    | a ∩ b |
    —————————
    | a ∪ b |
    """

    cdef int i, n = len(a), union_ = 0, intersection = 0
    cdef double ratio
    for i in prange(n, nogil=True):
        union_ += a[i] | b[i]
        intersection += a[i] & b[i]
    ratio = intersection / <double>union_
    return ratio


# faster than pyobj but slower than numba
def jaccard_similarity_cython(a, b):
    return _jaccard_similarity(a, b)


def jaccard_similarity_pyobj(cnp.ndarray a, cnp.ndarray b):
    """
    Return the ratio of the intersection to the union of two containers.

    | a ∩ b |
    —————————
    | a ∪ b |
    """
    cdef cnp.ndarray intersection = a & b, union_ = a | b
    cdef double ratio = intersection.sum() / union_.sum()
    return ratio


def sigmoid_pyobj(double x):
    """Sigmoid function defined as 1 / (1 + exp(-x))."""
    return 1 / (1 + exp(-x))


cdef double _sigmoid(double x):
    return 1 / (1 + exp(-x))

# same as pyobj speed
def sigmoid_cython(x):
    return _sigmoid(x)


@boundscheck(False)
@wraparound(False)
def cohesion(cnp.ndarray chrom, cnp.ndarray doc):
    """Measure of how compact all the clusters are.
    
     k            sim(Si, Sj)
     ∑      ∑     ———————————
    p=1  Si,Sj∈Cp    |Cp|
    """

    cdef:
        double total = 0.0
        int i, j, k, p
        int[:, ::1] sents
    
    for p in np.unique(chrom):
        sents = doc[chrom == p]
        k = len(sents)
        #: itertools.combinations(sents, r=2)
        for i in prange(k-1, nogil=True):
            for j in range(i+1, k):
                total += _jaccard_similarity(sents[i], sents[j]) / k
    return total



@boundscheck(False)
@wraparound(False)
def separation(cnp.ndarray chrom, cnp.ndarray doc):
    """Measure of how separable all the clusters are.

    k-1     k                 sim(Si, Sj)
     ∑      ∑      ∑      ∑   ———————————
    p=1   q=p+1  Si∈Cp  Sj∈Cq  |Cp|·|Cp|
    """

    cdef:
        double total = 0
        int i, j, k, m, n, p, q = len(np.unique(chrom))
        int[:, ::1] sents_p, sents_q
    
    #: itertools.combinations(k, r=2)
    for p in range(k-1):
        for q in range(p+1, k):
            sents_p = doc[chrom == p]
            sents_q = doc[chrom == q]
            #: itertools.product(sents_p, sents_q)
            m, n = len(sents_p), len(sents_q)
            for i in prange(m, nogil=True):
                for j in range(n):
                    total += _jaccard_similarity(sents_p[i], sents_q[j]) / m / n
    return total