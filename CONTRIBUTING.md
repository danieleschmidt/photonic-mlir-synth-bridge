# Contributing to Photonic MLIR

We welcome contributions to the Photonic MLIR project! This document provides guidelines for contributing to the codebase.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Contributions](#submitting-contributions)
- [Research Contributions](#research-contributions)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.9+
- LLVM/MLIR 17+
- CMake 3.18+
- Git

### Setup Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danieleschmidt/photonic-mlir-synth-bridge.git
   cd photonic-mlir-synth-bridge
   ```

2. **Run the setup script:**
   ```bash
   ./scripts/setup.sh
   ```

3. **Activate the development environment:**
   ```bash
   source venv/bin/activate
   source .env
   ```

4. **Verify installation:**
   ```bash
   python -c "import photonic_mlir; print('Setup successful!')"
   ```

## Development Workflow

### Branching Strategy

We use a Git Flow-inspired branching model:

- `main`: Stable production code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation branches

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Follow the coding standards** (see [Code Style](#code-style))
2. **Write tests** for new functionality
3. **Update documentation** as needed
4. **Commit frequently** with clear messages

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `build`: Build system changes
- `ci`: CI/CD changes

Examples:
```bash
git commit -m "feat(compiler): add wavelength multiplexing optimization"
git commit -m "fix(validation): handle edge case in power budget validation"
git commit -m "docs(api): update PhotonicCompiler documentation"
```

## Code Style

### Python Code Style

We use the following tools for Python code:

- **Formatter**: [Black](https://black.readthedocs.io/)
- **Linter**: [Flake8](https://flake8.pycqa.org/)
- **Type Checker**: [MyPy](https://mypy.readthedocs.io/)
- **Import Sorter**: [isort](https://pycqa.github.io/isort/)

#### Configuration

The project includes configuration files:
- `pyproject.toml`: Black, isort, MyPy configuration
- `.flake8`: Flake8 configuration

#### Running Code Style Checks

```bash
# Format code
black python/

# Sort imports
isort python/

# Lint code
flake8 python/

# Type check
mypy python/
```

### C++ Code Style

For MLIR/C++ code:

- Use LLVM coding standards
- 2-space indentation
- 80-character line limit
- Comprehensive documentation

### Pre-commit Hooks

Install pre-commit hooks to automatically check code style:

```bash
pre-commit install
```

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── performance/    # Performance tests
└── fixtures/       # Test data and fixtures
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run tests with coverage
python -m pytest --cov=photonic_mlir --cov-report=html

# Run performance tests (slower)
python -m pytest tests/performance/ --runslow
```

### Writing Tests

#### Unit Tests

```python
import pytest
from photonic_mlir import PhotonicCompiler, PhotonicBackend

class TestPhotonicCompiler:
    def test_compiler_creation(self):
        compiler = PhotonicCompiler(
            backend=PhotonicBackend.SIMULATION_ONLY,
            wavelengths=[1550.0, 1551.0],
            power_budget=100.0
        )
        assert compiler.backend == PhotonicBackend.SIMULATION_ONLY
        assert len(compiler.wavelengths) == 2
```

#### Integration Tests

```python
@pytest.mark.integration
def test_end_to_end_compilation(sample_model, sample_input):
    compiler = PhotonicCompiler()
    circuit = compiler.compile(sample_model, sample_input)
    assert circuit is not None
    assert circuit.config["power_budget"] > 0
```

#### Performance Tests

```python
@pytest.mark.performance
@pytest.mark.slow
def test_compilation_performance(large_model):
    start_time = time.time()
    compiler = PhotonicCompiler()
    circuit = compiler.compile(large_model, sample_input)
    compilation_time = time.time() - start_time
    
    assert compilation_time < 30.0  # Should compile within 30 seconds
```

### Test Markers

Use pytest markers to categorize tests:

- `@pytest.mark.unit`: Unit tests (fast)
- `@pytest.mark.integration`: Integration tests (medium)
- `@pytest.mark.performance`: Performance tests (slow)
- `@pytest.mark.hardware`: Hardware-dependent tests
- `@pytest.mark.slow`: Slow tests (require `--runslow` flag)

## Documentation

### Types of Documentation

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Step-by-step tutorials
3. **Developer Guides**: Architecture and design documentation
4. **Examples**: Working code examples

### Writing Docstrings

Use Google-style docstrings:

```python
def validate_wavelengths(wavelengths: List[float], pdk: str = "AIM_Photonics_PDK") -> Dict[str, Any]:
    """Validate wavelength specifications for photonic compilation.
    
    This function checks that the provided wavelengths are within valid ranges
    for the specified Process Design Kit (PDK) and ensures proper spacing
    to avoid crosstalk.
    
    Args:
        wavelengths: List of wavelength values in nanometers
        pdk: Process Design Kit name for validation rules
        
    Returns:
        Dictionary containing validation results:
            - valid: Boolean indicating if wavelengths are valid
            - count: Number of wavelength channels
            - range: Tuple of (min, max) wavelengths
            
    Raises:
        ValidationError: If wavelengths are invalid or improperly spaced
        
    Example:
        >>> result = validate_wavelengths([1550.0, 1551.0])
        >>> assert result["valid"] == True
        >>> assert result["count"] == 2
    """
```

### Building Documentation

```bash
# Generate API documentation
./scripts/setup.sh docs

# View documentation
open docs/_build/html/index.html
```

## Submitting Contributions

### Pull Request Process

1. **Create a feature branch** from `develop`
2. **Make your changes** following the guidelines above
3. **Test thoroughly** including unit, integration, and performance tests
4. **Update documentation** as needed
5. **Submit a pull request** to the `develop` branch

### Pull Request Template

When creating a PR, use this template:

```markdown
## Description
Brief description of the changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] New tests added for new functionality
- [ ] Performance tests pass (if applicable)

## Checklist
- [ ] Code follows the project style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Documentation updated
- [ ] Changes generate no new warnings

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Additional Notes
Any additional information about the changes.
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and style checks
2. **Peer Review**: At least one maintainer reviews the code
3. **Testing**: Changes are tested in various environments
4. **Approval**: PR is approved and merged to `develop`

## Research Contributions

We particularly welcome research contributions that advance the state of photonic neural network compilation:

### Research Areas

1. **Novel Photonic Algorithms**: New algorithms optimized for photonic hardware
2. **Optimization Techniques**: Advanced compilation optimizations
3. **Hardware Abstractions**: Support for new photonic devices
4. **Performance Analysis**: Benchmarking and comparative studies

### Research Contribution Process

1. **Discuss Your Idea**: Open an issue to discuss your research direction
2. **Literature Review**: Ensure your contribution is novel
3. **Implementation**: Implement your research with proper baselines
4. **Evaluation**: Provide comprehensive evaluation and analysis
5. **Documentation**: Write detailed research documentation
6. **Submission**: Submit as a PR with research documentation

### Research Standards

- **Reproducibility**: All experiments must be reproducible
- **Baselines**: Compare against relevant baselines
- **Statistical Significance**: Use proper statistical analysis
- **Documentation**: Provide detailed methodology and results

Example research contribution structure:

```
research/
└── wavelength_optimization/
    ├── README.md              # Research description
    ├── methodology.md         # Detailed methodology
    ├── experiments/          # Experimental code
    ├── results/              # Results and analysis
    ├── benchmarks/           # Benchmark comparisons
    └── publication/          # Paper or documentation
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

### Communication

- **GitHub Issues**: Bug reports, feature requests, discussions
- **GitHub Discussions**: General questions and community discussions
- **Email**: team@photonic-mlir.org for private matters

### Getting Help

- **Documentation**: Check the official documentation first
- **Search Issues**: Look for existing solutions in GitHub issues
- **Ask Questions**: Open a discussion or issue for help
- **Community**: Engage with the community for support

### Recognition

Contributors are recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributors mentioned in releases
- **Academic Papers**: Research contributors as co-authors (when applicable)

## Development Tips

### Useful Commands

```bash
# Run specific tests
python -m pytest tests/test_compiler.py::TestPhotonicCompiler::test_compile

# Debug test failures
python -m pytest -vv -s --tb=long

# Profile performance
python -m pytest --profile-svg tests/performance/

# Check test coverage
python -m pytest --cov=photonic_mlir --cov-report=html
open htmlcov/index.html
```

### IDE Setup

#### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- GitLens
- C/C++ (for MLIR development)

#### PyCharm

1. Configure interpreter to use `venv/bin/python`
2. Enable Black formatter
3. Configure MyPy as external tool

### Debugging

```python
# Add to code for debugging
import pdb; pdb.set_trace()

# Or use breakpoint() in Python 3.7+
breakpoint()
```

### Performance Profiling

```bash
# Profile Python code
python -m cProfile -o profile.stats your_script.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Memory profiling
pip install memory-profiler
python -m memory_profiler your_script.py
```

## FAQ

### Q: How do I add support for a new photonic operation?
A: 1) Define the operation in the MLIR dialect, 2) Implement the Python frontend, 3) Add compilation support, 4) Write tests and documentation.

### Q: How do I optimize compilation performance?
A: Profile your code, use caching effectively, consider parallel processing, and optimize algorithmic complexity.

### Q: How do I add a new hardware backend?
A: Implement the hardware interface, add device-specific optimizations, create simulation models, and provide thorough testing.

### Q: How do I contribute to the MLIR dialect?
A: Follow LLVM/MLIR contribution guidelines, ensure proper ODS definitions, implement verifiers, and add comprehensive tests.

---

Thank you for contributing to Photonic MLIR! Your contributions help advance the field of photonic AI accelerators and benefit the entire research community.