import math

import pytest

from evolvesumm import rouge


def test_rouge_n():
    extracted_summ = "The quick brown fox jumps over the lazy dog."
    gold_summ = "The quick fox jumps over the dog."
    assert math.isclose(rouge.rouge_n(1, extracted_summ, gold_summ), (6 / 6))  #: set removes 'the' duplication
    assert math.isclose(rouge.rouge_n(2, extracted_summ, gold_summ), (4 / 6))
    assert math.isclose(rouge.rouge_n(3, extracted_summ, gold_summ), (2 / 5))
    assert math.isclose(rouge.rouge_n(4, extracted_summ, gold_summ), (1 / 4))
