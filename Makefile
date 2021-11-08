update-deps:
	pip install pip-tools
	pip-compile --generate-hashes --output-file=requirements.txt dev.in

format:
	black . --line-length=120
