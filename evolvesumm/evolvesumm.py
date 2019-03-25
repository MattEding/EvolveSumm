import functools
import itertools

from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
# from sklearn.metrics import pairwise_distances


import dde
import similarity


def fitness_func(chromosome, doc, func):
    return func(chromosome, similarity.jaccard_similarity, doc)


def evolesumm(text, summary_length=0.1, population_size=100, iterations=1000, lambda_=0.5,
              crossover_rate=0.5, *, distance=cosine_distances, fitness='coh_sep', verbose=False, seed=None):

    fit_funcs = {'coh_sep': similarity.cohesion_separation, 'cohesion': similarity.cohesion,
                 'separation': similarity.separation}
    func = fit_funcs[fitness]

    cv = CountVectorizer(stop_words='english')
    sents_lower = tokenize.sent_tokenize(text.lower())
    sents_lower = (sent.split('\n') for sent in sents_lower)
    sents_lower = tuple(itertools.chain.from_iterable(sents_lower))
    vec = cv.fit_transform(sents_lower)
    doc = vec.toarray().astype(bool).astype(int)

    fitness2 = functools.partial(fitness_func, func=func, doc=doc)

    best = dde.main(population_size=population_size, summary_length=summary_length,
                    sentence_count=len(doc), fitness=fitness2, lambda_=lambda_,
                    crossover_rate=crossover_rate, iterations=iterations,
                    verbose=verbose, seed=None)

    summ = dde.construct_summary(best, doc, text, sents_lower, distance, fitness2)
    return summ
