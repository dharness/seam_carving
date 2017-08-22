clean-pyc:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf

test:
	python -m unittest tests.seam_carver_test