all: lint test install

lint:
	env/bin/flake8 hmm/*.py tests/*.py

test:
	env/bin/nosetests tests

install:
	env/bin/pip install -e .

notebook:
	env/bin/jupyter notebook --notebook-dir=docs
