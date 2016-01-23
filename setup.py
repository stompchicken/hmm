try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Hidden Markov Models',
    'author': 'stompchicken',
    'url': 'github.com/stompchicken/hmm',
    'download_url': 'github.com/stompchicken/hmm',
    'author_email': 'steve@stompchicken.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['hmm'],
    'scripts': [],
    'name': 'hmm'
}

setup(**config)
