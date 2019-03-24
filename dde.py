import functools
import itertools
import logging
import multiprocessing
import pathlib

import numba
import numpy as np

from nltk import tokenize
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances

import similarity


def init_chromosome(choices, length):
    chrom = np.full(length, -1)
    #: ensure that each choice is accounted for at least once
    idxs = np.random.choice(np.arange(length), len(choices), replace=False)
    chrom[idxs] = np.random.permutation(choices)
    idxs = np.where(chrom == -1)[0]
    chrom[idxs] = np.random.choice(choices, len(idxs))
    return chrom


def init_population(population_size, cluster_amount, chromosome_length):
    clusters = np.arange(cluster_amount)
    chroms = [init_chromosome(clusters, chromosome_length) for _ in range(population_size)]
    population = np.vstack(chroms)
    return population


def get_offspring(population, randoms, lambda_, crossover_rate):
    n = np.arange(len(population))
    s = set(n)
    idxs = np.array([np.random.choice(tuple(s - {i}), size=3, replace=False) for i in n])
    chrom_1, chrom_2, chrom_3 = map(np.squeeze, np.split(population[idxs], 3, axis=1))
    k = len(np.unique(population))
    offspr = (chrom_1 + lambda_ * (chrom_2 - chrom_3)) % k
    mask = randoms < crossover_rate
    offspr[mask] = population[mask]
    return offspr


def next_generation(population, offspring, fitness, pool):
    fits = pool.map(fitness, itertools.chain(population, offspring))
    i = len(population)
    fit_pop = np.array(fits[:i])
    fit_off = np.array(fits[i:])
    mask = fit_off > fit_pop
    population[mask] = offspring[mask]
    return None


def mutate(population, randoms):
    mask = randoms < similarity.sigmoid(population)
    #: inversion operator
    idxs = np.nonzero(mask)
    arr = np.array(idxs)
    sorter = np.lexsort((-arr[1], arr[0]))
    rev = arr.T[sorter].T
    population[idxs] = population[(rev[0], rev[1])]
    return None


def iterate_gerations(population_size, summary_length, sentence_count, fitness,
                      lamba_, crossover_rate, iterations):
    pool = multiprocessing.Pool()
    pop = init_population(population_size, summary_length, sentence_count)
    shape = pop.shape
    for i in range(iterations):
        rand = np.random.random_sample(shape)
        offspr = get_offspring(pop, rand, lamba_, crossover_rate)
        next_generation(pop, offspr, fitness, pool)
        mutate(pop, rand)
        yield pop
    pool.terminate()  #TODO: will this die with yields?


def best_chromosome(population, fitness):
    #TODO: make sure it picks one with all k-clusters
    fits = np.argmax([fitness(chrom) for chrom in population])
    chrom = population[fits]
    return chrom


def central_sentences(chromosome, document, metric):
    central_sents = []
    for cluster in np.unique(chromosome):
        idxs = np.where(chromosome == cluster)[0]
        sents = document[idxs]
        centroid = sents.mean(axis=0)[np.newaxis,:]
        dists = metric(sents, centroid)
        cent_sent = idxs[np.argmin(dists)]
        central_sents.append(cent_sent)
    return sorted(central_sents)


def retrieve_orig(idxs, orig_text, tokens):
    summ_evol = []
    for sent in np.array(tokens)[idxs]:
        start = orig_text.lower().index(sent)
        stop = start + len(sent)
        summ_evol.append(orig_text[start:stop])
    summ_evol = '\n'.join(summ_evol)
    return summ_evol


def construct_summary(population, document, orig_text, tokens, metric, fitness):
    chrom = best_chromosome(population, fitness)
    central = central_sentences(chrom, document, metric)
    summ = retrieve_orig(central, orig_text, tokens)
    return summ


def main(population_size, summary_length, sentence_count, fitness, lamba_,
         crossover_rate, iterations, *, seed=None, verbose=False):

    if verbose:
        logging.info(f'population_size={population_size}, summary_length={summary_length},'
                     f'fitness={fitness}, lamba_={lamba_}, crossover_rate={crossover_rate},'
                     f'iterations={iterations}, seed={seed}')

    if seed is not None:
        np.random.seed(seed)

    if isinstance(summary_length, int):
        if not (0 < summary_length < sentence_count):
            raise ValueError('int summary_length must be between 0 and the number of sentences in the document')
    elif isinstance(summary_length, float):
        if not (0.0 < summary_length < 1.0):
            raise ValueError('float summary_length must be between 0.0 and 1.0')
        summary_length = int(summary_length * sentence_count) or 1
    else:
        raise TypeError('summary_length must be a float or int')

    for i, generation in enumerate(iterate_gerations(population_size, summary_length,
                                                     sentence_count, fitness, lamba_,
                                                     crossover_rate, iterations)):
        if verbose:
            logging.info(f'iteration: {i}')

    # summ = construct_summary(generation, document, orig_text, tokens, )
    return generation
