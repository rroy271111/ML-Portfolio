.PHONY: train api test lint

train:
	@echo "Training model..."
	python -m src.models.itrain --config configs/default.yaml

api:
	echo "Starting API service on port 8000..."
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

test:
	@echo "Running tests..."
	pytest -q --disable-warnings

lint:
	@echo "Linting code..."
	flake8 src tests

clean:
	@echo "Cleaning up cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytes_cache