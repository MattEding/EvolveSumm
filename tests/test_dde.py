from evolvesumm import dde

import pytest


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


def test_create_offspring():
    ...


def test_central_tokens():
    ...


def test_build_summary():
    ...


def test_early_stopping():
    ...


def test_verbose():
    ...


