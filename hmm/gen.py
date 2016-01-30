import numpy as np


def normalise(x):
    """Helper function, curiously absent in numpy"""
    return x.astype(float) / np.sum(x)


def states(N):
    """Generate N hidden states"""
    return ['z%d' % i for i in range(N)]


def symbols(M):
    """Generate M emission symbols"""
    return ['y%d' % i for i in range(M)]


# Generation functions for distributions

# Thse functions take only one argument: the dimensionality of the (row)
# vector to return

def uniform_dist(N):
    """Uniform distribution over N items"""
    return np.repeat(1.0 / N, N)


def sparse_uniform_dist(N, p=0.5):
    """Uniform distribution over random size pN subset of N items"""
    return normalise(np.random.binomial(1, p, size=N) / np.sum())


def dirichlet_dist(N, k=1):
    """Distribution over N items sampled from symmetric Dirichlet"""
    return np.random.dirichlet([k for i in range(N)])


def lognormal_dist(N, mean=0, sigma=1):
    """Distribution over N items randomly drawn from a log-normal"""
    return normalise(np.random.lognormal(mean, sigma, size=N))


def zipf_dist(N, a=2):
    """Disribution over N items randomly drawn from a Zipfian"""
    return normalise(np.random.zipf(a, size=N))
