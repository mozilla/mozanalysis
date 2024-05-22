dev_dependencies:
	pip install -r requirements-dev.txt
	pip install -e .

test_dependencies:
	pip install -r requirements.txt
	pip install -e .

update_deps: dev_dependencies
	./script/update_deps