import numba
import numpy as np

from nltk import tokenize
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances


@numba.njit
def jaccard_similarity(a, b):
    union = np.sum(a | b)
    if not union:
        return 1.0
    return np.sum(a & b) / union


@numba.njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@numba.njit(cache=True)
def get_sentences(doc, chromosome, k):
    return doc[chromosome == k]


@numba.njit
def cohesion(chromosome, similarity, document):
    total = 0
    for p in np.unique(chromosome):
        # sents = document[chromosome == p]
        sents = get_sentences(doc, chromosome, p)
        k = len(sents)
        #: combinations choose 2
        for i in range(k-1):
            for j in range(i+1, k):
                total += similarity(sents[i], sents[j]) / len(sents)
    return total


@numba.njit
def separation(chromosome, similarity, document):
    total = 0
    k = len(np.unique(chromosome))
    #: combinations choose 2
    for p in range(k-1):
        for q in range(p+1, k):
            sents_p = get_sentences(doc, chromosome, p)
            sents_q = get_sentences(doc, chromosome, q)
            # sents_p = document[chromosome == p]
            # sents_q = document[chromosome == q]
            #: product
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


def init_chromosome(choices, length):
    chrom = np.full(length, -1)
    #: ensure that each choice is accounted for at least once
    idxs = np.random.choice(np.arange(length), len(choices), replace=False)
    chrom[idxs] = np.random.permutation(choices)
    idxs = np.where(chrom == -1)[0]
    chrom[idxs] = np.random.choice(choices, len(idxs))
    return chrom


def init_population(population_size, cluster_amount, chromosome_length):
    clusts = np.arange(cluster_amount)
    chroms = [init_chromosome(clusts, chromosome_length) for _ in range(population_size)]
    pop = np.vstack(chroms)
    return pop


def get_offspring(population, randoms, lambda_, crossover_rate):
    #: For computation time, relax requirement that X_r, X_r1, X_r2, X_r3 are distinct.
    #: With large population size, this is unlikely to occur, and if it does, it doesn't
    #: seem that detrimental. Also is this mitigated with appropriate lam choice?
    n = len(population)
    idxs = np.random.choice(np.arange(n), size=(n, 3))
    chrom_1, chrom_2, chrom_3 = map(np.squeeze, np.split(population[idxs], 3, axis=1))
    k = len(np.unique(population))
    offspr = (chrom_1 + lambda_ * (chrom_2 - chrom_3)) % k
    mask = randoms < crossover_rate
    offspr[mask] = population[mask]
    return offspr


def next_generation(population, offspring, func):
    fit_off = np.array([func(chrom) for chrom in offspring])
    fit_pop = np.array([func(chrom) for chrom in population])
    mask = fit_off > fit_pop
    population[mask] = offspring[mask]
    return


def mutate(population, randoms):
    mask = randoms < sigmoid(population)
    #: inversion operator
    idxs = np.nonzero(mask)
    arr = np.array(idxs)
    sorter = np.lexsort((-arr[1], arr[0]))
    rev = arr.T[sorter].T
    population[idxs] = population[(rev[0], rev[1])]
    return


#TODO: early stopping --> little fitness improvement over x generations, good enough fitness score
def run_iterations(pop_size, summ_len, num_sents, func, lam, cr, iterations, *, mutate_after=True,
                   seed=None, verbose=False, save_rate=np.nan, save_dir=None):

    if save_dir is not None:
        save_dir = pathlib.Path(save_dir)
        if not save_dir.is_dir():
            msg = f'save_dir={save_dir} not a valid directory path'.format(save_dir=save_dir)
            raise NotADirectoryError(msg)

    if seed is not None:
        np.random.seed(seed)

    pop = init_population(pop_size, summ_len, num_sents)
    shape = pop.shape
    for i in range(iterations):
        if i % save_rate == 0:
            file = save_dir / 'generation_{i:0>pad}'.format(i=i, pad=len(str(iterations)))
            np.save(file, pop)

        if verbose:
            print(i)  #TODO: logfile --> iteration number, best fitness score, avg fitness score, hyper-params

        rand = np.random.random_sample(shape)
        offspr = get_offspring(pop, rand, lam, cr)
        #: option since papers unclear if mutate at offspring or survivors stage
        if not mutate_after:
            mutate(offspr, rand)

        next_generation(pop, offspr, func)
        if mutate_after:
            mutate(pop, rand)
    return pop


def best_chromosome(population):
    #TODO: make sure it picks one with all k-clusters
    fits = np.argmax([fitness(chrom) for chrom in population])
    chrom = population[fits]
    return chrom


def central_sentences(chromosome, document, metric=cosine_distances):
    central_sents = []
    for cluster in np.unique(chromosome):
        idxs = np.where(chromosome == cluster)[0]
        sents = document[idxs]
        centroid = sents.mean(axis=0)[np.newaxis,:]
        dists = metric(sents, centroid)
        cent_sent = idxs[np.argmin(dists)]
        central_sents.append(cent_sent)
    return sorted(central_sents)


if __name__ == '__main__':
    import json
    import pathlib
    import time


    jsons = pathlib.Path.cwd() / 'data' / 'jsons'
    json_2018 = jsons / '2018' / '2018.json'

    with open(json_2018) as fp:
        articles_2018 = json.load(fp)['2018']

    article = articles_2018[222]
    text = article['story']


    def fitness(chromosome):
        return cohesion_separation(chromosome, jaccard_similarity, doc)

    cv = CountVectorizer(stop_words='english')
    sents_lower = tokenize.sent_tokenize(text.lower())
    vec = cv.fit_transform(sents_lower)
    doc = vec.toarray().astype(bool).astype(int)

    t0 = time.time()
    pop = run_iterations(pop_size=100, summ_len=5, num_sents=len(doc),
                         func=fitness, lam=0.9, cr=0.5, iterations=300, verbose=False, seed=0)
    t1 = time.time()
    print(t1-t0)
