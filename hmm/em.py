import numpy as np

# TODO: Work in log space


def decode(model, symbols, initial_state=None):
    """Viterbi algorithm.
    Given an HMM in a certain state and a sequence of
    emission symbols, what is the most likely state for each symbol?
    """

    # Map symbols to internal indexes
    y = [model.symbol_to_index(symbol) for symbol in symbols]
    T = len(y)

    # V[z][i] is the the probability of the most likely path to z_i=z
    V = np.zeros((model.N, T))
    # bp is the last most likely state given current state
    # bp[z][i] = argmax(p(z_i-1|z_i=z))
    bp = np.zeros((model.N, T), dtype=np.int)

    # Initialise the hidden state
    V[:, 0] = model._initial[:] * model._emission[y[0], :]

    for i in range(1, T):
        # z1 the previous state, z2 is current state
        for z2 in range(model.N):

            # state, probability
            z, p_z = 0, 0.0

            # Consider all possible previous states
            for z1 in range(model.N):
                p = V[z1][i - 1]
                p *= model._transition[z1][z2]
                p *= model._emission[y[i]][z2]

                # Keep track of max probability
                if p > p_z:
                    z, p_z = z1, p

            # Update with probability from most likely previous state
            V[z2][i] = p_z
            bp[z2][i] = z

    z = []
    # Take the highest probability final state
    last_state = int(np.argmax(V[:, T - 1]))
    z.append(last_state)
    # Work backwards...
    for i in range(1, len(y)):
        last_state = int(bp[last_state][T - i])
        z.append(last_state)
    z.reverse()

    # Map state indexes to states
    Z = [model.index_to_state(z_i) for z_i in z]
    return Z


def forward(model, y, a):
    """Forward pass. Calculate forward matrix based on model parameters"""
    T = len(y)
    # Calculate recursively, from the beginning of the sequence to the end
    a[:, 0] = model._initial * model._emission[:, y[0]]
    for i in range(1, T):
        a[:, i] = np.dot(a[:, i - 1], model._transition) * model._emission[:, y[i]]
    return a


def backward(model, y, b):
    """Backwards pass. Calculate backwards matrix based on model parameters"""
    T = len(y)
    # Calculate recursively, from the end of the sequence to the beginning
    b[:, T - 1] = 1
    for i in reversed(range(0, T - 2)):
        b[:, i] = np.dot(b[:, i + 1], model._transition[i, :]) * model._emission[:, y[i]]
    return b

def baum_welch_init(model, y):
    T = len(y)

    # Construct intermediate matrices
    # Forward matrix: a[i,t] = p(z_t=i|y[0:t-1])
    a = np.zeros((model.N, T))
    # Backward matrix: b[i,t] = p(z_t=i|y[t+1:T])
    b = np.zeros((model.N, T))
    # gamma[z, i] = p(z_i=z)
    gamma = np.zeros((model.N, T))
    # xi[z1, z2, i] = p(z_i=z1, z_i+1=z2)
    xi = np.zeros((model.N, model.N, T))

    return (a, b, gamma, xi)

def baum_welch_update(model, y, a, b, gamma, xi):
    T = len(y)

    # Update forward and backward matices
    forward(model, y, a)
    backward(model, y, b)

    # Calculate intermediate matrices
    gamma = a * b
    gamma /= np.sum(gamma, axis=0, keepdims=True)

    # Broadcast all matrices to shape (N, N, M)
    xi = a[np.newaxis, :, :] \
         * model._emission.T[np.newaxis, :, :] \
         * model._transition[:, :, np.newaxis] \
         * b[np.newaxis, :, :]
    xi /= np.sum(a[:,T - 1])

    # Initial distribution is first column of gamma
    model._initial = gamma[:, 0]

    # Sum to t-1???
    #model._transition = np.sum(xi, axis=2) / np.sum(gamma, axis=1)
    #model._emission = np.sum(gamma, axis=1) / np.sum(gamma, axis=1)
    

def learn(model, symbols):
    """Baum-Welch algorithm.
    """
    # Map symbols to internal indexes
    y = [model.symbol_to_index(symbol) for symbol in symbols]

    a, b, gamma, xi = baum_welch_init(model, y)

    for i in range(1):
        baum_welch_update(model, y, a, b, gamma, xi)
