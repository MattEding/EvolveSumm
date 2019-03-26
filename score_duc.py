import itertools
import json
import logging
import pathlib

import gensim
from nltk import tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer

import dde
import process_duc
from rouge import rouge_n
import similarity


cwd = pathlib.Path.cwd()
data = cwd / 'data'
duc = data / 'duc'
files = sorted(duc.iterdir())

logfile = cwd / f'articles.log'
logfile.touch()
fmt = '{name} - {asctime} - {levelname} : {message}'
logging.basicConfig(filename=logfile, level=logging.INFO, style='{', format=fmt)


def fitness(chromosome):
    return similarity.cohesion_separation(chromosome, similarity.jaccard_similarity, doc)


dde_scores = []
gs_scores = []
tr_scores = []
lr_scores = []
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
    
    original = original.replace('\n', ' ')
    abstract = abstract.replace('\n', ' ')
    
    cv = CountVectorizer(stop_words='english')
    sents_lower = tokenize.sent_tokenize(original.lower())
    sents_lower = (sent.split('\n') for sent in sents_lower)
    sents_lower = tuple(itertools.chain.from_iterable(sents_lower))
    vec = cv.fit_transform(sents_lower)
    doc = vec.toarray().astype(bool).astype(int)

    # dde
    best = dde.main(population_size=100, summary_length=0.1, sentence_count=len(doc), fitness=fitness, lamba_=0.5,
                    crossover_rate=0.5, iterations=1000)
    
    dde_summ = dde.construct_summary(best, doc, original, sents_lower, cosine_distances, fitness)
    dde_summ.replace('\n', ' ')
    dde_rouge = rouge_n(1, dde_summ, abstract), rouge_n(2, dde_summ, abstract), rouge_n(3, dde_summ, abstract)
    dde_scores.append(dde_rouge)
    
    # gensim
    gs_summ = gensim.summarization.summarize(original, ratio=0.1)
    gs_rouge = rouge_n(1, gs_summ, abstract), rouge_n(2, gs_summ, abstract), rouge_n(3, gs_summ, abstract)
    gs_scores.append(gs_rouge)
    
    # sumy
    num_sents = 0.1 * len(doc) or 1
    parser = PlaintextParser(original, Tokenizer('english'))
    
    text_rank = TextRankSummarizer()
    tr_summ = text_rank(parser.document, num_sents)
    tr_summ = ' '.join(str(s) for s in tr_summ)
    tr_rouge = rouge_n(1, tr_summ, abstract), rouge_n(2, tr_summ, abstract), rouge_n(3, tr_summ, abstract)
    tr_scores.append(tr_rouge)
    
    lex_rank = LexRankSummarizer()
    lr_summ = lex_rank(parser.document, num_sents)
    lr_summ = ' '.join(str(s) for s in lr_summ)
    lr_rouge = rouge_n(1, lr_summ, abstract), rouge_n(2, lr_summ, abstract), rouge_n(3, lr_summ, abstract)
    lr_scores.append(lr_rouge)
    
with open('dde_scores.json', 'w') as fp:
    json.dump(dde_scores, fp)

with open('gs_scores.json', 'w') as fp:
    json.dump(gs_scores, fp)

with open('tr_scores.json', 'w') as fp:
    json.dump(tr_scores, fp)
    
with open('lr_scores.json', 'w') as fp:
    json.dump(lr_scores, fp)

logging.info('saved dde, gensim, textrank, lexrank scores')