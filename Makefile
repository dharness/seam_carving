.PHONY: demo


clean-pyc:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf

test:
	python -m unittest tests.seam_carver_test

demo:
	python -m unittest tests.visual_output_test