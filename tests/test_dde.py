import itertools

import numpy as np
import pytest

from evolvesumm import dde


@pytest.fixture(scope='module')
def dde_summ():
    dde_summ = dde.DdeSummarizer()
    dde_summ.fit('...')
    return dde_summ


# can i chain from result of one test to another to continue from
def test_fit():
    ...


def test_mutate():
    ...


def test_initialize_chromosome():
    ...


def test_survival():
    ...


def test_get_distinct_chromosomes():
    dde_summ = dde.DdeSummarizer()
    dde_summ._pop = np.random.random_sample(size=(20, 5))
    chrom_1, chrom_2, chrom_3 = dde_summ._get_distinct_chromosomes()
    combos = itertools.combinations([dde_summ._pop, chrom_1, chrom_2, chrom_3], r=2)
    assert not np.any([np.any(chrom_x == chrom_y, axis=1) for chrom_x, chrom_y in combos])


def test_chromosomal_inversion():
    mask = np.array([[ True,  True, False],
                     [ True, False,  True],
                     [False,  True, False],
                     [ True,  True,  True]])
    pop = np.arange(mask.size).reshape(mask.shape)
    inv = dde.DdeSummarizer._chromosomal_inversion(mask, pop.copy())
    assert np.all(inv == np.array([[ 1,  0,  2],
                                   [ 5,  4,  3],
                                   [ 6,  7,  8],
                                   [11, 10,  9]]))


def test_central_tokens():
    ...


def test_build_summary():
    ...


def test_early_stopping():
    ...


def test_verbose():
    ...


