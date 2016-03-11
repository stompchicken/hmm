all: lint build

lint:
	flake8 hmm/*.py tests/*.py

build:
	env/bin/nosetests tests
