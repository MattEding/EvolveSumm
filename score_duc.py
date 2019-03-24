import itertools
import json
import logging
import pathlib

import gensim
from nltk import tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

import dde
import process_duc
from rouge import rouge_n
import similarity


cwd = pathlib.Path.cwd()
data = cwd / 'data'
duc = data / 'duc'
files = list(duc.iterdir())

logfile = cwd / f'articles.log'
logfile.touch()
fmt = '{name} - {asctime} - {levelname} : {message}'
logging.basicConfig(filename=logfile, level=logging.INFO, style='{', format=fmt)


def fitness(chromosome):
    return similarity.cohesion_separation(chromosome, similarity.jaccard_similarity, doc)


dde_scores = []
tr_scores = []
for i, file in enumerate(files):
    np.random.seed(i)
    logging.info(f'article {i}')
    try:
        abstract, original = process_duc.extract(file)
        if len(abstract.split()) < 10 or len(original.split()) < 10:
            logging.info(f'skipping article {i}')
            continue
    except StopIteration:
        logging.info(f'skipping article {i}')
        continue
    
    cv = CountVectorizer(stop_words='english')
    sents_lower = tokenize.sent_tokenize(original.lower())
    sents_lower = (sent.split('\n') for sent in sents_lower)
    sents_lower = tuple(itertools.chain.from_iterable(sents_lower))
    vec = cv.fit_transform(sents_lower)
    doc = vec.toarray().astype(bool).astype(int)

    best = dde.main(population_size=100, summary_length=0.1, sentence_count=len(doc), fitness=fitness, lamba_=0.5,
                    crossover_rate=0.5, iterations=1000)
    
    

    dde_summary = dde.construct_summary(best, doc, original, sents_lower, cosine_distances, fitness)
    dde_rouge = rouge_n(1, dde_summary, abstract), rouge_n(2, dde_summary, abstract), rouge_n(3, dde_summary, abstract)
    dde_scores.append(dde_rouge)
    
    tr_summary = gensim.summarization.summarize(original, ratio=0.1)
    tr_rouge = rouge_n(1, tr_summary, abstract), rouge_n(2, tr_summary, abstract), rouge_n(3, tr_summary, abstract)
    tr_scores.append(tr_rouge)
    
    
with open('dde_scores.json', 'w') as fp:
    json.dump(dde_scores, fp)
with open('tr_scores.json', 'w') as fp:
    json.dump(tr_scores, fp)
logging.info('saved dde and tr scores')