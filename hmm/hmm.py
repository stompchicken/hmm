import numpy as np


def is_prob_density(p):
    """Does the given matrix have columns that all sum to one?"""
    return np.sum(np.sum(p, axis=0)) == p.shape[1]


class HiddenMarkovModel(object):

    def __init__(self, states, symbols):
        """
        Creates an HMM with N hidden states and M emission symbols.
        All transition and emission probabilities are created equal
        """
        if len(states) == 0:
            raise ValueError("Must have at least one state")

        if len(states) != len(set(states)):
            raise ValueError("Cannot have duplicate states")

        if len(symbols) == 0:
            raise ValueError("Must have at least one symbol")

        if len(symbols) != len(set(symbols)):
            raise ValueError("Cannot have duplicate symbols")

        self.states = states
        self.symbols = symbols
        self.N = len(states)
        self.M = len(symbols)

        self._initial = np.ones((self.N))

        # transition[i, j] = p(z2=j|z1=i)
        self._transition = np.ones((self.N, self.N))

        # emission[i,j] -> p(y=i,z=j)
        self._emission = np.ones((self.M, self.N))
        self.normalise()

    def zero(self):
        """Reset probabilities to zero"""
        self._initial = np.zeros(self._initial.shape)
        self._transition = np.zeros(self._transition.shape)
        self._emission = np.zeros(self._emission.shape)

    def is_valid(self):
        return np.sum(self._initial) == 1.0 \
            and is_prob_density(self._transition) \
            and is_prob_density(self._emission)

    def normalise(self):
        """Normalise probability distributions to sum to unity"""
        self._initial /= np.sum(self._initial, axis=0, keepdims=True)
        self._transition /= np.sum(self._transition, axis=0, keepdims=True)
        self._emission /= np.sum(self._emission, axis=0, keepdims=True)

    def state_to_index(self, state):
        """Map hidden state to internal index"""
        for i, s in enumerate(self.states):
            if s == state:
                return i
        return -1

    def index_to_state(self, i):
        """Map internal index to hidden state"""
        return self.states[i]

    def symbol_to_index(self, symbol):
        """Map emission symbol to internal index"""
        for i, s in enumerate(self.symbols):
            if s == symbol:
                return i
        return -1

    def index_to_symbol(self, i):
        """Map internal index to emission symbol"""
        return self.symbols[i]

    def initial(self):
        """Return dictionary mapping initial hidden states to probabilties"""
        return dict(zip(self.states, self._initial.tolist()))

    def set_initial(self, values):
        """Set initial distribution from dictionary of states to probabilities"""
        for key, value in values.items():
            i = self.state_to_index(key)
            self._initial[i] = value

    def transition(self, state):
        """Return dictionary mapping next hidden states to probabilties"""
        i = self.state_to_index(state)
        return dict(zip(self.states, self._transition[:, i].tolist()))

    def set_transition(self, state, values):
        """Set transition distribution for a state from dictionary of states to probabilities"""
        i = self.state_to_index(state)
        for key, value in values.items():
            j = self.state_to_index(key)
            self._transition[i, j] = value

    def emission(self, state):
        """Return dictionary mapping emission symbols to probabilties"""
        i = self.state_to_index(state)
        return dict(zip(self.symbols, self._emission[:, i].tolist()))

    def set_emission(self, state, values):
        """Set emission distribution for a state from dictionary of symbols to probabilities"""
        i = self.state_to_index(state)
        for key, value in values.items():
            j = self.symbol_to_index(key)
            self._emission[j, i] = value

    def __repr__(self):
        return "<%s %d states, %d symbols>" % (self.__class__.__name__, self.N, self.M)


def create_from_data(states, symbols, tagged_data):
    """
    Takes a sequence of (state, symbol) tuples and two lists of all
    states and symbols
    """
    model = HiddenMarkovModel(list(states), list(symbols))

    # Reset probabilities to zero
    model.zero()

    # Count observed transitions and emissions
    for seq in tagged_data:
        for i, (tag, word) in enumerate(seq):
            z = model.state_to_index(tag)
            y = model.symbol_to_index(word)

            if i == 0:
                model._initial[z] += 1.0
                model._emission[y][z] += 1.0
            else:
                z1 = model.state_to_index(seq[i - 1][0])
                model._transition[z1][z] += 1.0
                model._emission[y][z] += 1.0

    model.normalise()
    return model


def create_from_random(states, symbols, transition_dist, emission_dist):
    """
    Create a random HMM by sampling rows of the transition and
    emission matrices from the given random generation functions.
    """
    model = HiddenMarkovModel(list(states), list(symbols))
    N, M = len(states), len(symbols)

    # Reset probabilities to zero
    model.zero()

    # Initial probabilities
    model._initial = transition_dist(N)

    # Transition probabilities
    for z in range(N):
        model._transition[:, z] = transition_dist(N)

    # Emission probabilities
    for y in range(N):
        model._emission[:, y] = emission_dist(M)

    return model
