import numpy as np

cimport numpy as cnp
from cython cimport boundscheck, wraparound
from cython.parallel cimport prange
from libc.math cimport exp



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

    for i in range(n):
        union_ += a[i] | b[i]
        intersection += a[i] & b[i]
    ratio = intersection / <double>union_
    return ratio


cdef double _sigmoid(double x):
    return 1 / (1 + exp(-x))


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
        int i, j, k = len(np.unique(chrom)), m, n, p, q = len(np.unique(chrom))
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


def cohesion_separation(cnp.ndarray chrom, cnp.ndarray doc):
    """Measure balancing both cohesion and separation of clusters."""
    cdef double coh, sep
    coh = cohesion(chrom, doc)
    sep = separation(chrom, doc)
    return (1 + _sigmoid(coh)) ** sep


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# For quick testing & benchmarking in ipython
#
# NOTES: for some reason casting as int32 (int) is 4x faster than uint8 (unsigned char);
#           2x faster than int8 (char);
#           no np.bool_ memview support;
#           at least the int version is 4x faster than my numba version! :D
#           have to manually cast doc from bool to int32
#
# GOAL: parallelize the multiprocessing 
#           use pointers on flatten versions of array and keep track of row lengths
#           for doc[chrom == k] can loop thru memoryview once and store clusters in mapping
#               is there a defaultdict(list) for c/c++?
#               want to store as map[cluster].append(sentence index)
#               **ptr to do the trick?

from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import bitarray


def get_document(text):
    stop_words = None
    text = str(text)
    tokens = sent_tokenize(text.lower())
    count_vec = CountVectorizer(stop_words=stop_words).fit_transform(tokens)
    #: numba does not support sparse matrices; dtype bool to emulate sets
    document = count_vec.toarray().astype(bool)
    return document


def get_chromosome(document):
    summ_len = 5
    #: make a chromosome that is a random partition with each cluster.
    clusters = np.arange(summ_len)
    chrom = np.full(len(document), -1)
    #: ensure that each cluster is accounted for at least once
    idxs = np.random.choice(np.arange(len(chrom)), summ_len, replace=False)
    chrom[idxs] = np.random.permutation(clusters)
    #: fill rest randomly
    idxs = (chrom == -1)
    chrom[idxs] = np.random.choice(clusters, np.sum(idxs))
    return chrom


with open('/Users/matteding/Desktop/Metis/EvolveSumm/poe.txt') as fp:
    text = fp.read()

np.random.seed(0)
doc = get_document(text).astype(np.int32)
chrom = get_chromosome(doc)
