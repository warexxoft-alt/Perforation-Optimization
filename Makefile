.PHONY: help install install-dev run test test-cov lint lint-fix format docker-build docker-run clean

help:
	@echo "PerfoSim Pro v3 — developer commands"
	@echo "  make install       Install runtime dependencies"
	@echo "  make install-dev   Install runtime + dev/test/lint dependencies"
	@echo "  make run           Launch the Streamlit app locally"
	@echo "  make test          Run the pytest suite"
	@echo "  make test-cov      Run pytest with coverage report"
	@echo "  make lint          Check code style with ruff"
	@echo "  make lint-fix      Auto-fix style issues with ruff"
	@echo "  make docker-build  Build the production Docker image"
	@echo "  make docker-run    Run the app in a container on :8501"
	@echo "  make clean         Remove caches and build artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	streamlit run app.py

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov --cov-report=term-missing --cov-report=html

lint:
	ruff check .

lint-fix:
	ruff check . --fix

docker-build:
	docker build -t perfosim-pro:latest .

docker-run:
	docker run --rm -p 8501:8501 perfosim-pro:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
