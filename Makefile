.PHONY: demo


clean-pyc:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf

lint:
	python -m pylint ./seam_carver.py

# When using the 'only' option, you must prefix
# the test name with a '.' (period)
# e.g. make test only=.TestSeamCarver.test_add_seam
test:
	python -m unittest tests.seam_carver_test$(only)

demo:
	python -m unittest tests.visual_output_test