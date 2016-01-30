from nose.tools import assert_equals

import hmm
import numpy as np


def test_hmm_default():
    model = hmm.HiddenMarkovModel(["1", "2"], ["A", "B", "C", "D"])
    assert model
    assert model.is_valid()

    assert_equals(model.state_to_index("1"), 0)
    assert_equals(model.state_to_index("2"), 1)
    assert_equals(model.symbol_to_index("A"), 0)
    assert_equals(model.symbol_to_index("B"), 1)
    assert_equals(model.symbol_to_index("C"), 2)
    assert_equals(model.symbol_to_index("D"), 3)

    assert_equals(model.initial(), {"1": 0.5, "2": 0.5})
    assert_equals(model.transition("1"), {"1": 0.5, "2": 0.5})
    assert_equals(model.transition("2"), {"1": 0.5, "2": 0.5})
    assert_equals(model.emission("1"), {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    assert_equals(model.emission("2"), {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})


def test_create_from_data():
    states = ["1", "2"]
    symbols = ["A", "B", "C", "D"]
    tagged_data = [[("1", "A"), ("2", "C"), ("1", "A")],
                   [("1", "B"), ("2", "D")]]
    model = hmm.create_from_data(states, symbols, tagged_data)
    assert(model)
    assert(model.is_valid())


def test_create_from_random():
    states = ["1", "2"]
    symbols = ["A", "B", "C", "D"]

    def transition_dist(N):
        return np.ones(N) / N

    def emission_dist(N):
        return np.ones(N) / N

    model = hmm.create_from_random(states, symbols, transition_dist, emission_dist)
    assert(model)
    assert(model.is_valid())
