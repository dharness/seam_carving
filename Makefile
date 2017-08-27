.PHONY: demo


clean-pyc:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf

lint:
	python -m pylint ./seam_carver.py

test:
	python seam_carver/seam_carver_test.py

build:
	python setup.py sdist bdist_wheel

clean-build:
	rm -rf dist
	rm -rf build
	rm -rf seam_carver.egg-info

publish:
	twine upload dist/*