from nose.tools import assert_equals

import hmm
import models

def test_em_decode():
    model = models.flip_flop()
    assert model.is_valid()

    states = hmm.decode(model, ["A", "B", "A", "B"])
    assert_equals(4, len(states))
    assert_equals(["1", "2", "1", "2"], states)


def test_em_learn():
    model = hmm.HiddenMarkovModel(["1", "2"], ["A", "B"])
    data = [["A", "B", "A", "B"]]
    hmm.learn(model, data)
    assert model.is_valid()
