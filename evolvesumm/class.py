import collections
import logging
import reprlib

from nltk.tokenize import sent_tokenize
from numba import njit
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from utils import jaccard, sigmoid


class DdeSummarizer:
    """Discrete differential evolution (DDE) extractive text summarizer.

    Parameters
    ----------
    pop_size : int, (default=1000)
        The population size to create offspring and mutate over.

    max_iter : int, (default=1000)
        The maximum number of generations to evolve.

    summ_ratio : float, (default=0.1)
        The compression ratio for the summary, with 0 <= summ_ratio <= 1.

    lam : float, (default=0.5)
        The scale factor used with DDE, with 0 <= lam <= 1.

    crossover : float, (default=0.5)
        The crossover rate used for offspring, with 0 <= summ_ratio <= 1.

    fitness : str, 'coh_sep', 'coh', or 'sep', (default='coh_sep')
        The fitness function used to determine which chromosomes make it to the
        next generation. The 'coh' fitness maximizes similarity within a given
        cluster. The 'sep' fitness minimizes similarity between different
        clusters. The 'coh_sep' is a balance of the former two.

    similarity : callable, (default=jaccard)
        Similarity function for comparing two arrays. Needs to be able to work
        with numba.

    metric : str, or callable, (default='cosine')
        Metric used to select central sentences from clusters when finished with
        iterating through generations.
        See sklearn.metrics.pairwise_distances for details.

    tokenizer : callable, (default=nltk.tokenize.sent_tokenize)
        Tokenizer used to split text when fit.

    stop_words : str, list, or None (default=None):
        Words to remove from document.
        See sklearn.feature_extraction.text.CountVectorizer for details.

    n_jobs : int, (default=1)
        The number of CPUs to score fitness of each chromosome in the population
        at each generation. -1 means usings all processors.

    early_stopping : bool, (default=False)
        Whether to use early stopping to terminate iterations when fitness score
        is not improving.

    n_iter_no_change : int, (default=5)
        Number of iterations with no improvement to wait before early stopping.

    tol : float, (default=1e-3)
        The stopping criterion.

    random_state : int, (default=None)
        The seed of the pseudo random number generator to use when evolving To
        be passed to np.random.seed.

    verbose : int, (default=0)
        The verbosity level.
    """

    def __init__(self, pop_size=100, max_iter=1000, summ_ratio=0.1, lam=0.5, crossover=0.5,
                 fitness='coh_sep', similarity=jaccard, metric='cosine', tokenizer=sent_tokenize,
                 stop_words=None, n_jobs=1, early_stopping=False, n_iter_no_change=5,
                 tol=1e-3, random_state=None, verbose=0):

        self.max_iter = int(max_iter)
        self.n_jobs = int(n_jobs) # if negative -> use all cores
        self.early_stopping = bool(early_stopping)
        self.n_iter_no_change = int(n_iter_no_change) if (n_iter_no_change > 1) else 1
        self.tol = float(tol)
        self.random_state = random_state
        self.verbose = max(0, int(verbose))
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.metric = str(metric).lower()

        self._pop = None
        self._offspr = None
        self._rand = None
        self._pool = None

        funcs = dict(coh_sep=self._cohesion_separation, coh=self._cohesion, sep=self._separation)
        self.fitness = funcs[fitness.lower()]

        if (summ_ratio < 0) or (summ_ratio > 1):
            raise ValueError('summ_ratio not in interval [0, 1]')
        self.summ_ratio = float(summ_ratio)

        if (lam < 0) or (lam > 1):
            raise ValueError('lam not in interval [0, 1]')
        self.lam = float(lam)

        if (crossover < 0) or (crossover > 1):
            raise ValueError('crossover not in interval [0, 1]')
        self.crossover = float(crossover)

        #TODO: use inspect.signature to see if it takes 2 inputs
        if not callable(similarity):
            raise ValueError('similarity must be callable')
        self.similarity = similarity

    def __repr__(self):
        fitness = '_'.join(fit[:3] for fit in self.fitness.__name__.split('_') if fit)
        return f'{type(self).__name__}(text={reprlib.repr(text)}, ...TODO)'

    def __next__(self):
        self._rand = np.random.random_sample(self._pop.shape)
        self._offspring()
        self._survival()
        self._mutate()

    def fit(self, text):
        self.text = str(text)
        self._tokens = self.tokenizer(self.text.lower())
        count_vec = CountVectorizer(stop_words=self.stop_words).fit_transform(self._tokens)
        #: numba does not support sparse matrices
        self._document = count_vec.toarray().astype(bool)
        self._summ_len = int(self.summ_ratio * self._document.shape[1]) or 1

    def summarize(self):
        np.random.seed(self.random_state)

        if self.verbose:
            logging.debug(repr(self))
            logging.info(self.text)
            logging.debug('random seed: {}'.format(self.random_state))
            if self.verbose >= 2:
                logging.info('random state: {}'.format(np.random.get_state()))

        processes = self.n_jobs if (self.n_jobs >= 1) else None
        self._pool = multiprocessing.Pool(processes)
        self._pop = np.array([self._init_chrom() for _ in range(self.pop_size)])

        n_iter = collections.deque([np.nan] * self.n_iter_no_change, maxlen=self.n_iter_no_change)
        for i in range(self.max_iter):
            self._rand = np.random.random_sample(self._pop.shape)
            next(self)
            if self.verbose:
                logging.debug('iteration: {}'.format(i))
                if self.verbose >= 2:
                    logging.debug('best fit: {}'.format(self._best_fit))

            if self.early_stopping:
                n_iter.append(self._best_fit)
                if max(n_iter) - min(n_iter) < self.tol:
                    break

        self._pool.terminate()
        self.n_iter_ = i + 1
        #TODO: this skips mutate step?
        self.best_chrom_ = self._best_fit

    def _init_chrom(self):
        choices = np.arange(self._summ_len)
        chrom = np.full(self._document.shape[1], -1)
        #: ensure that each choice is accounted for at least once
        idxs = np.random.choice(np.arange(len(chrom)), self._summ_len, replace=False)
        chrom[idxs] = np.random.permutation(choices)
        #: fill rest randomly
        idxs = (chrom == -1)
        chrom[idxs] = np.random.choice(choices, np.sum(idxs))
        return chrom

    def _offspring(self):
        n = np.arange(len(self._pop))
        s = frozenset(n)
        #: get 3 distinct chromosomes that differ from ith chromosome
        idxs = np.array([np.random.choice(tuple(s - {i}), size=3, replace=False) for i in n])
        chrom_1, chrom_2, chrom_3 = map(np.squeeze, np.split(self._pop[idxs], 3, axis=1))
        #: discrete differential evolution
        self._offspr = (chrom_1 + self.lam * (chrom_2 - chrom_3)) % self._summ_len
        mask = self._rand < self.crossover
        self._offspr[mask] = self._pop[mask]
        return

    def _survival(self):
        fits = self._pool.map(self.fitness, itertools.chain(self._pop, self._offspr))
        self._best_fit = fits.max()
        fit_pop, fit_off = np.split(fits, 2)
        mask = fit_off > fit_pop
        self._pop[mask] = self._offspr[mask]
        return

    def _mutate(self):
        mask = self._rand < sigmoid(self._pop)
        #: inversion operator -> for each row reverse order of all True values
        idxs = np.nonzero(mask)
        arr = np.array(idxs)
        sorter = np.lexsort((-arr[1], arr[0]))
        rev = arr.T[sorter].T
        self._pop[idxs] = self._pop[(rev[0], rev[1])]
        return

    def _central_sents(self):
        central_idxs = []
        for cluster in np.unique(self.best_chrom_):
            idxs = np.where(self.best_chrom_ == cluster)[0]
            sents = self._document[idxs]
            centroid = sents.mean(axis=0)[np.newaxis,:]
            dists = pairwise_distances(sents, centroid, self.metric)
            cent_sent = idxs[np.argmin(dists)]
            central_idxs.append(cent_sent)
        return sorted(central_idxs)

    def _build_summ(self):
        central = self._central_sents()
        summ = []
        for sent in np.array(self._tokens)[central]:
            start = self.text.lower().index(sent)
            stop = start + len(sent)
            summ.append(self.text[start:stop])
        self.summary_ = '\n'.join(summ)
        return

    @njit
    def _cohesion(self, chrom):
        total = 0
        for p in np.unique(chrom):
            sents = self._document[chrom == p]
            k = len(sents)
            #: itertools.combinations(sents, r=2)
            for i in range(k-1):
                for j in range(i+1, k):
                    total += self.similarity(sents[i], sents[j]) / len(sents)
        return total

    @njit
    def _separation(self, chrom):
        total = 0
        k = len(np.unique(chrom))
        #: itertools.combinations(k, r=2)
        for p in range(k-1):
            for q in range(p+1, k):
                sents_p = self._document[chrom == p]
                sents_q = self._document[chrom == q]
                #: itertools.product(sents_p, sents_q)
                m, n = len(sents_p), len(sents_q)
                for i in range(m):
                    for j in range(n):
                        total += self.similarity(sents_p[i], sents_q[j]) / m / n
        return total

    @njit
    def _cohesion_separation(self, chrom):
        coh = self._cohesion(chrom)
        sep = self._separation(chrom)
        return (1 + sigmoid(coh)) ** sep
