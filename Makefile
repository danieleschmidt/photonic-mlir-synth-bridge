# Photonic MLIR Makefile
# Provides common development and build tasks

.PHONY: help build clean install test lint format docs dev-install

# Default target
help:
	@echo "Photonic MLIR Development Commands:"
	@echo ""
	@echo "Build Commands:"
	@echo "  build          - Build MLIR libraries and Python bindings"
	@echo "  build-release  - Build optimized release version"
	@echo "  build-debug    - Build debug version with symbols"
	@echo "  clean          - Clean build artifacts"
	@echo ""
	@echo "Python Commands:"
	@echo "  install        - Install Python package in development mode"
	@echo "  dev-install    - Install with development dependencies"
	@echo "  uninstall      - Uninstall Python package"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-performance - Run performance tests"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-mlir      - Run MLIR dialect tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Run all linters"
	@echo "  format         - Format code with black and isort"
	@echo "  mypy           - Run type checking"
	@echo "  security       - Run security checks"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           - Build documentation"
	@echo "  docs-serve     - Serve documentation locally"
	@echo ""
	@echo "Development:"
	@echo "  setup-dev      - Set up development environment"
	@echo "  pre-commit     - Install pre-commit hooks"
	@echo "  benchmark      - Run benchmarks"

# Build configuration
BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Release
PYTHON_VERSION ?= $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Build targets
build:
	@echo "Building Photonic MLIR..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -G Ninja .. \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DMLIR_DIR=$$(llvm-config --prefix)/lib/cmake/mlir \
		-DLLVM_EXTERNAL_LIT=$$(llvm-config --prefix)/bin/llvm-lit
	cd $(BUILD_DIR) && ninja

build-release:
	@$(MAKE) build CMAKE_BUILD_TYPE=Release

build-debug:
	@$(MAKE) build CMAKE_BUILD_TYPE=Debug

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf python/photonic_mlir.egg-info
	rm -rf python/build
	rm -rf python/dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete

# Python installation
install:
	@echo "Installing Photonic MLIR Python package..."
	pip install -e .

dev-install:
	@echo "Installing Photonic MLIR with development dependencies..."
	pip install -e ".[dev,docs,benchmark]"

uninstall:
	@echo "Uninstalling Photonic MLIR..."
	pip uninstall photonic-mlir -y

# Testing
test:
	@echo "Running all tests..."
	pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	pytest tests/ -v -m unit

test-integration:
	@echo "Running integration tests..."
	pytest tests/ -v -m integration

test-performance:
	@echo "Running performance tests..."
	pytest tests/ -v -m performance --runslow

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=photonic_mlir --cov-report=html --cov-report=term-missing

test-mlir:
	@echo "Running MLIR dialect tests..."
	cd $(BUILD_DIR) && ninja check-photonic-dialect

# Code quality
lint:
	@echo "Running linters..."
	flake8 python/photonic_mlir tests/
	black --check python/photonic_mlir tests/
	isort --check-only python/photonic_mlir tests/
	mypy python/photonic_mlir

format:
	@echo "Formatting code..."
	black python/photonic_mlir tests/
	isort python/photonic_mlir tests/

mypy:
	@echo "Running type checking..."
	mypy python/photonic_mlir

security:
	@echo "Running security checks..."
	bandit -r python/photonic_mlir -f json
	safety check

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && make html

docs-serve:
	@echo "Serving documentation at http://localhost:8000..."
	cd docs/_build/html && python -m http.server 8000

# Development setup
setup-dev:
	@echo "Setting up development environment..."
	python -m pip install --upgrade pip setuptools wheel
	pip install -r requirements-dev.txt
	pre-commit install

pre-commit:
	@echo "Installing pre-commit hooks..."
	pre-commit install
	pre-commit install --hook-type commit-msg

# Benchmarking
benchmark:
	@echo "Running benchmarks..."
	python -m pytest tests/ -v -m performance --benchmark-only

# Docker targets
docker-build:
	@echo "Building Docker image..."
	docker build -t photonic-mlir:latest .

docker-test:
	@echo "Running tests in Docker..."
	docker run --rm photonic-mlir:latest make test

# CI/CD helpers
ci-install:
	@echo "Installing for CI..."
	pip install -e ".[dev]"

ci-test:
	@echo "Running CI tests..."
	pytest tests/ -v --junit-xml=test-results.xml --cov=photonic_mlir --cov-report=xml

ci-lint:
	@echo "Running CI linting..."
	flake8 python/photonic_mlir tests/ --output-file=flake8-report.txt
	black --check python/photonic_mlir tests/
	isort --check-only python/photonic_mlir tests/
	mypy python/photonic_mlir --junit-xml=mypy-report.xml

# Release helpers
version-check:
	@echo "Current version:"
	@grep version setup.py | head -1

release-check:
	@echo "Checking release readiness..."
	python -m build --check
	twine check dist/*

# Environment info
env-info:
	@echo "Environment Information:"
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "PyTorch version: $(shell python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Build type: $(CMAKE_BUILD_TYPE)"

# Maintenance
update-deps:
	@echo "Updating dependencies..."
	pip-compile requirements.in
	pip-compile requirements-dev.in

check-deps:
	@echo "Checking for security vulnerabilities..."
	safety check
	@echo "Checking for outdated packages..."
	pip list --outdated