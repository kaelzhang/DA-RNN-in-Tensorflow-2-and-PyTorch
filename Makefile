files = da_rnn test *.py
test_files = *

test:
	pytest -s -v test/test_$(test_files).py --doctest-modules --cov da_rnn --cov-config=.coveragerc --cov-report term-missing

lint:
	flake8 $(files)

fix:
	autopep8 --in-place -r $(files)

install:
	pip install -U -r requirements.txt -r dev-requirements.txt

report:
	codecov

build: da_rnn
	rm -rf dist
	python setup.py sdist bdist_wheel

publish:
	make build
	twine upload --config-file ~/.pypirc -r pypi dist/*

.PHONY: test build
