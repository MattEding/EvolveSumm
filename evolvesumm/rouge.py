from nltk.util import ngrams


def rouge_n(n, y_pred, y_true):
    n_gram_pred = set(ngrams(y_pred, n))
    n_gram_true = set(ngrams(y_true, n))
    return len(n_gram_pred & n_gram_true) / len(n_gram_true)
