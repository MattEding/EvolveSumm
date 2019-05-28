import math

import numpy as np
import pytest

from evolvesumm import utils


def test_jaccard():
    arr_1 = np.array([1, 0, 0, 1, 1])
    arr_2 = np.array([1, 1, 0, 0, 0])
    arr_zeros = np.zeros(5, dtype=int)
    assert math.isclose(utils.jaccard(arr_1, arr_2), 0.25)
    assert utils.jaccard(arr_1, arr_1) == 1.0
    assert utils.jaccard(arr_zeros, arr_zeros) == 1.0


def test_sigmoid():
    ...


def test_cohesion():
    ...


def test_separation():
    ...


def test_cohesion_separation():
    ...
