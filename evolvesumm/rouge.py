import string

from nltk.tokenize import word_tokenize
from nltk.util import ngrams


def _get_ngrams(text, n):
    punctuation = set(string.punctuation)
    no_punc = "".join(char for char in text.lower() if char not in punctuation)
    words = word_tokenize(no_punc)
    return set(ngrams(words, n))


def rouge_n(n, extracted_summ, gold_summ):
    """ROUGE-N metric for evaluating how good a summary is to a gold summary
    (e.g. human made summary). It is the number of matchin n-grams divided by
    the total n-grams in gold summary.

    This implementation is derived from:
    [1] C.Y. Lin "ROUGE: A Package for Automatic Evaluation of Summaries" 2004
    """
    
    n_gram_pred = _get_ngrams(extracted_summ, n)
    n_gram_true = _get_ngrams(gold_summ, n)
    return len(n_gram_pred & n_gram_true) / len(n_gram_true)


#TODO: implement other rouge score variations
