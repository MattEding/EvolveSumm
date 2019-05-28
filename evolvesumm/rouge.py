import string

from nltk.util import ngrams


def sanitize(text):
    no_punctation = "".join(char for char in text if char not in string.punctuation)
    return no_punctation.lower().split()


def rouge_n(n, extracted_summ, gold_summ):
    """ROUGE-N metric for evaluating how good a summary is to a gold summary
    (e.g. human made summary). It is the number of matchin n-grams divided by
    the total n-grams in gold summary.

    This implementation is derived from:
    [1] C.Y. Lin "ROUGE: A Package for Automatic Evaluation of Summaries" 2004
    """
    n_gram_pred = set(ngrams(sanitize(extracted_summ), n))
    n_gram_true = set(ngrams(sanitize(gold_summ), n))
    return len(n_gram_pred & n_gram_true) / len(n_gram_true)


#TODO: implement other rouge score variations
