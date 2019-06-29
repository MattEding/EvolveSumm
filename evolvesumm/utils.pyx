import numpy as np

from cython cimport boundscheck, wraparound
from cython.parallel cimport prange
from libc.math cimport exp
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


ctypedef vector[int[::1]] int_vec_t
ctypedef unordered_map[int, int_vec_t] int_vec_map_t

ctypedef int[::1] chrom_t

ctypedef int[:, ::1] doc_t
ctypedef int[::1] sent_t # same type as doc; just 1D


@boundscheck(False)
@wraparound(False)
cdef inline double _jaccard_similarity(sent_t a, sent_t b) nogil:
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
    if union_ == 0:
        union_ = 1
    ratio = intersection / <double>union_
    return ratio


# for numpy broadcasting
def sigmoid(x):
    """Sigmoid function defined as 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-x))


@boundscheck(False)
@wraparound(False)
cdef inline double _cohesion(chrom_t chrom, doc_t doc) nogil:
    cdef:
        double total = 0.0
        int i, j, k, cluster, length = len(chrom), 
        sent_t sent
        int_vec_t sents
        int_vec_map_t sent_map

    for i in range(length):
        cluster = chrom[i]
        sent = doc[i]
        sent_map[cluster].push_back(sent)

    for pair in sent_map:
        sents = pair.second
        k = sents.size()
        #: itertools.combinations(sents, r=2)
        for i in prange(k - 1):
            for j in range(i + 1, k):
                total += _jaccard_similarity(sents[i], sents[j]) / k
    return total


def cohesion(chrom, doc):
    """Measure of how compact all the clusters are.

     k            sim(Si, Sj)
     ∑      ∑     ———————————
    p=1  Si,Sj∈Cp    |Cp|
    """
    return _cohesion(chrom, doc)


@boundscheck(False)
@wraparound(False)
cdef inline double _separation(chrom_t chrom, doc_t doc) nogil:
    cdef:
        double total = 0.0
        int i, j, k, m, n, p, q, cluster, length = len(chrom)
        sent_t sent
        int_vec_t sents_p, sents_q
        int_vec_map_t sent_map

    for i in range(length):
        cluster = chrom[i]
        sent = doc[i]
        sent_map[cluster].push_back(sent)
    
    k = sent_map.size()
    #: itertools.combinations(k, r=2)
    for p in range(k - 1):
        for q in range(p + 1, k):
            sents_p = sent_map[p]
            sents_q = sent_map[q]
            m = sents_p.size()
            n = sents_q.size()
            #: itertools.product(sents_p, sents_q)
            for i in range(m):
                for j in range(n):
                    total += _jaccard_similarity(sents_p[i], sents_q[j]) / m / n
    return total


def separation(chrom, doc):
    """Measure of how separable all the clusters are.

    k-1     k                 sim(Si, Sj)
     ∑      ∑      ∑      ∑   ———————————
    p=1   q=p+1  Si∈Cp  Sj∈Cq  |Cp|·|Cp|
    """
    return _separation(chrom, doc)


cdef inline double _sigmoid(double x) nogil:
    return 1 / (1 + exp(-x))


cdef double _cohesion_separation(chrom_t chrom, doc_t doc) nogil:
    cdef double coh, sep
    coh = _cohesion(chrom, doc)
    sep = _separation(chrom, doc)
    return (1 + _sigmoid(coh)) ** sep


def cohesion_separation(chrom, doc):
    """Measure balancing both cohesion and separation of clusters."""
    return _cohesion_separation(chrom, doc)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# For quick testing & benchmarking in ipython
#
# NOTES: for some reason casting as int32 (int) is 4x faster than uint8 (unsigned char);
#           2x faster than int8 (char);
#           no np.bool_ memview support;
#           at least the int version is 4x faster than my numba version! :D
#           have to manually cast doc from bool to int32
#
# TODO: 
#   - use fuse types on chromosomes
#   - manage sparse matrices properly
#   - change DDE _survival to take advantage of nogil

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

np.random.seed(1)
doc = get_document(text).astype(np.int32)
chrom = get_chromosome(doc).astype(np.int32)
