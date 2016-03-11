import hmm


def flip_flop():
    """HMM flip-flopping between two states"""
    model = hmm.HiddenMarkovModel(["1", "2"], ["A", "B"])

    model.set_initial({"1": 1.0, "2": 0.0})

    model.set_transition("1", {"1": 0.0, "2": 1.0})
    model.set_transition("2", {"1": 1.0, "2": 0.0})

    model.set_emission("1", {"A": 1.0, "B": 0.0})
    model.set_emission("2", {"A": 0.0, "B": 1.0})

    return model
