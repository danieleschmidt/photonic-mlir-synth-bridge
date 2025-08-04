# CI/CD Pipeline Configuration

Due to GitHub permissions requirements, the CI/CD workflow needs to be manually created. Here's the complete workflow configuration that should be placed in `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  PYTHON_VERSION: "3.11"
  LLVM_VERSION: "17"

jobs:
  # Linting and code quality
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements-dev.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        flake8 python/photonic_mlir tests/
        black --check python/photonic_mlir tests/
        isort --check-only python/photonic_mlir tests/
    
    - name: Run type checking
      run: mypy python/photonic_mlir
    
    - name: Run security checks
      run: |
        bandit -r python/photonic_mlir -f json -o bandit-report.json
        safety check --json --output safety-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  # Unit and integration tests
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.9", "3.10", "3.11"]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install LLVM/MLIR (Ubuntu)
      if: matrix.os == 'ubuntu-latest'
      run: |
        wget https://apt.llvm.org/llvm.sh
        chmod +x llvm.sh
        sudo ./llvm.sh ${{ env.LLVM_VERSION }}
        sudo apt-get install -y libmlir-${{ env.LLVM_VERSION }}-dev mlir-${{ env.LLVM_VERSION }}-tools
    
    - name: Install LLVM/MLIR (macOS)
      if: matrix.os == 'macos-latest'
      run: |
        brew install llvm@${{ env.LLVM_VERSION }}
        echo "/opt/homebrew/opt/llvm@${{ env.LLVM_VERSION }}/bin" >> $GITHUB_PATH
    
    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-${{ matrix.python-version }}-pip-${{ hashFiles('requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-${{ matrix.python-version }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Build MLIR components
      run: |
        mkdir -p build
        cd build
        cmake -G "Unix Makefiles" .. \
          -DCMAKE_BUILD_TYPE=Release \
          -DMLIR_DIR=$(llvm-config-${{ env.LLVM_VERSION }} --prefix)/lib/cmake/mlir \
          -DLLVM_EXTERNAL_LIT=$(llvm-config-${{ env.LLVM_VERSION }} --prefix)/bin/llvm-lit
        make -j$(nproc)
    
    - name: Install Python package
      run: pip install -e .
    
    - name: Run unit tests
      run: pytest tests/ -v -m unit --junit-xml=test-results-unit.xml
    
    - name: Run integration tests
      run: pytest tests/ -v -m integration --junit-xml=test-results-integration.xml
    
    - name: Run MLIR tests
      run: |
        cd build
        make check-photonic-dialect
    
    - name: Generate coverage report
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == env.PYTHON_VERSION
      run: |
        pytest tests/ --cov=photonic_mlir --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == env.PYTHON_VERSION
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.os }}-${{ matrix.python-version }}
        path: |
          test-results-*.xml
          htmlcov/

  # Performance benchmarks
  benchmark:
    runs-on: ubuntu-latest
    needs: [test]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install LLVM/MLIR
      run: |
        wget https://apt.llvm.org/llvm.sh
        chmod +x llvm.sh
        sudo ./llvm.sh ${{ env.LLVM_VERSION }}
        sudo apt-get install -y libmlir-${{ env.LLVM_VERSION }}-dev mlir-${{ env.LLVM_VERSION }}-tools
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install -e ".[benchmark]"
    
    - name: Run benchmarks
      run: |
        pytest tests/ -v -m performance --benchmark-json=benchmark-results.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark-results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true

  # Build and test Docker images
  docker:
    runs-on: ubuntu-latest
    needs: [lint, test]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Docker Hub
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: photonicmlir/photonic-mlir
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: docker/Dockerfile
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Test Docker image
      run: |
        docker run --rm photonicmlir/photonic-mlir:latest python -c "import photonic_mlir; print('Docker image OK')"

  # Security scanning
  security:
    runs-on: ubuntu-latest
    needs: [docker]
    if: github.event_name != 'pull_request'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'photonicmlir/photonic-mlir:latest'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  # Documentation build
  docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install documentation dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Build documentation
      run: |
        sphinx-build -b html docs docs/_build/html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html

  # Release deployment
  release:
    runs-on: ubuntu-latest
    needs: [lint, test, docker, security]
    if: github.event_name == 'release'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build distribution packages
      run: python -m build
    
    - name: Check distribution packages
      run: twine check dist/*
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: twine upload dist/*
    
    - name: Create GitHub release assets
      run: |
        tar -czf photonic-mlir-${{ github.event.release.tag_name }}-source.tar.gz \
          --exclude='.git' --exclude='build' --exclude='*.pyc' .
    
    - name: Upload release assets
      uses: actions/upload-release-asset@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        upload_url: ${{ github.event.release.upload_url }}
        asset_path: ./photonic-mlir-${{ github.event.release.tag_name }}-source.tar.gz
        asset_name: photonic-mlir-${{ github.event.release.tag_name }}-source.tar.gz
        asset_content_type: application/gzip
```

## Manual Setup Instructions

1. **Create the workflow file**:
   - Go to your repository on GitHub
   - Navigate to `.github/workflows/`
   - Create a new file named `ci.yml`
   - Copy the content from above

2. **Configure secrets** (if needed):
   - `DOCKER_USERNAME` and `DOCKER_PASSWORD` for Docker Hub
   - `PYPI_TOKEN` for PyPI releases
   - `CODECOV_TOKEN` for code coverage

3. **Enable workflow permissions**:
   - Go to repository Settings → Actions → General
   - Under "Workflow permissions", select "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"

The CI/CD pipeline includes:
- ✅ Code linting and type checking
- ✅ Multi-platform testing (Ubuntu, macOS)  
- ✅ Multi-version Python support (3.9, 3.10, 3.11)
- ✅ MLIR compilation and testing
- ✅ Security scanning
- ✅ Docker image building
- ✅ Performance benchmarking
- ✅ Documentation building
- ✅ Automated releases