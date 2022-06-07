install: requirements.txt
	pip install -r requirements.txt
run:
	python3 -m sailingVLM discover .
run_tests:
	python3 -m unittest discover sailingVLM
clean:
	rm -rf __pycache__