#!/bin/bash
# Setup script for Photonic MLIR development and production environment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
LLVM_VERSION="${LLVM_VERSION:-17}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            OS="ubuntu"
        elif command -v yum &> /dev/null; then
            OS="centos"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        OS="unknown"
    fi
    
    log_info "Detected OS: $OS"
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    case $OS in
        ubuntu)
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                ninja-build \
                git \
                curl \
                wget \
                python${PYTHON_VERSION} \
                python${PYTHON_VERSION}-dev \
                python3-pip \
                python3-venv \
                clang-${LLVM_VERSION} \
                llvm-${LLVM_VERSION}-dev \
                libmlir-${LLVM_VERSION}-dev \
                mlir-${LLVM_VERSION}-tools \
                libeigen3-dev \
                libopenblas-dev
            ;;
        centos)
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                cmake \
                ninja-build \
                git \
                curl \
                wget \
                python${PYTHON_VERSION} \
                python${PYTHON_VERSION}-devel \
                python3-pip \
                clang \
                llvm-devel \
                openblas-devel
            ;;
        macos)
            if ! command -v brew &> /dev/null; then
                log_error "Homebrew not found. Please install Homebrew first."
                exit 1
            fi
            
            brew install \
                cmake \
                ninja \
                llvm@${LLVM_VERSION} \
                python@${PYTHON_VERSION} \
                eigen \
                openblas
            
            # Add LLVM to PATH for macOS
            echo 'export PATH="/opt/homebrew/opt/llvm@'${LLVM_VERSION}'/bin:$PATH"' >> ~/.zshrc
            echo 'export LDFLAGS="-L/opt/homebrew/opt/llvm@'${LLVM_VERSION}'/lib"' >> ~/.zshrc
            echo 'export CPPFLAGS="-I/opt/homebrew/opt/llvm@'${LLVM_VERSION}'/include"' >> ~/.zshrc
            ;;
        *)
            log_error "Unsupported OS: $OS"
            exit 1
            ;;
    esac
    
    log_success "System dependencies installed"
}

# Set up Python environment
setup_python() {
    log_info "Setting up Python environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        python${PYTHON_VERSION} -m venv venv
        log_info "Created virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    python -m pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi
    
    if [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
    fi
    
    log_success "Python environment set up"
}

# Build MLIR components
build_mlir() {
    log_info "Building MLIR components..."
    
    cd "$PROJECT_ROOT"
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure with CMake
    case $OS in
        ubuntu|centos)
            cmake -G Ninja .. \
                -DCMAKE_BUILD_TYPE=Release \
                -DMLIR_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/mlir \
                -DLLVM_EXTERNAL_LIT=/usr/bin/llvm-lit-${LLVM_VERSION}
            ;;
        macos)
            cmake -G Ninja .. \
                -DCMAKE_BUILD_TYPE=Release \
                -DMLIR_DIR=/opt/homebrew/opt/llvm@${LLVM_VERSION}/lib/cmake/mlir \
                -DLLVM_EXTERNAL_LIT=/opt/homebrew/opt/llvm@${LLVM_VERSION}/bin/llvm-lit
            ;;
    esac
    
    # Build
    ninja
    
    log_success "MLIR components built"
}

# Install Python package
install_package() {
    log_info "Installing Photonic MLIR package..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install in development mode
    pip install -e .
    
    log_success "Package installed"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run Python tests
    python -m pytest tests/ -v
    
    # Run MLIR tests (if available)
    if [[ -d "build" ]]; then
        cd build
        ninja check-photonic-mlir || log_warning "MLIR tests not available or failed"
    fi
    
    log_success "Tests completed"
}

# Set up development tools
setup_dev_tools() {
    log_info "Setting up development tools..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        log_info "Pre-commit hooks installed"
    fi
    
    # Set up VS Code settings if .vscode doesn't exist
    if [[ ! -d ".vscode" ]]; then
        mkdir -p .vscode
        
        # Create settings.json
        cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/build": true,
        "**/dist": true,
        "**/*.egg-info": true
    },
    "C_Cpp.default.includePath": [
        "\${workspaceFolder}/include",
        "/usr/include/llvm-c-${LLVM_VERSION}",
        "/usr/include/mlir-c-${LLVM_VERSION}"
    ]
}
EOF
        log_info "VS Code settings created"
    fi
    
    log_success "Development tools set up"
}

# Generate documentation
generate_docs() {
    log_info "Generating documentation..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install docs dependencies if not already installed
    pip install sphinx sphinx-rtd-theme myst-parser
    
    # Generate API documentation
    if [[ ! -d "docs" ]]; then
        mkdir docs
        cd docs
        sphinx-quickstart -q --sep -p "Photonic MLIR" -a "Photonic MLIR Team" -v "0.1.0" --ext-autodoc --ext-viewcode --makefile --no-batchfile .
    fi
    
    # Build documentation
    cd docs
    make html
    
    log_success "Documentation generated in docs/_build/html/"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Test imports
    python -c "
import photonic_mlir
print(f'Photonic MLIR version: {photonic_mlir.__version__}')

from photonic_mlir import PhotonicCompiler, PhotonicBackend
compiler = PhotonicCompiler(backend=PhotonicBackend.SIMULATION_ONLY)
print('✓ Core compiler functionality available')

from photonic_mlir import get_cache_manager, get_metrics_collector
cache = get_cache_manager()
metrics = get_metrics_collector()
print('✓ Caching and monitoring systems operational')

from photonic_mlir import create_local_cluster
cluster = create_local_cluster(num_nodes=2)
cluster.shutdown()
print('✓ Distributed compilation system operational')

print('🚀 Installation verification successful!')
"
    
    log_success "Installation verified successfully!"
}

# Clean build artifacts
clean() {
    log_info "Cleaning build artifacts..."
    
    cd "$PROJECT_ROOT"
    
    # Remove build directory
    if [[ -d "build" ]]; then
        rm -rf build
        log_info "Removed build directory"
    fi
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove egg-info
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Create development environment file
create_env_file() {
    log_info "Creating environment file..."
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# Photonic MLIR Environment Configuration

# Logging
PHOTONIC_LOG_LEVEL=INFO
PHOTONIC_DEBUG=false

# Caching
PHOTONIC_CACHE_ENABLED=true
PHOTONIC_CACHE_SIZE_MB=512
PHOTONIC_CACHE_DIR=./cache

# Compilation
PHOTONIC_MAX_WORKERS=4
PHOTONIC_MAX_CONCURRENT_JOBS=10
PHOTONIC_COMPILATION_TIMEOUT=300

# Security
PHOTONIC_ENABLE_SECURITY=true
PHOTONIC_JWT_SECRET=development-secret-change-in-production

# Hardware
PHOTONIC_ENABLE_GPU=false
PHOTONIC_GPU_MEMORY_FRACTION=0.8

# Development
PYTHONPATH=./python:\$PYTHONPATH
EOF
        log_info "Created .env file with default settings"
    else
        log_info ".env file already exists"
    fi
}

# Main setup function
setup() {
    log_info "Starting Photonic MLIR setup..."
    
    detect_os
    install_system_deps
    setup_python
    build_mlir
    install_package
    setup_dev_tools
    create_env_file
    run_tests
    verify_installation
    
    log_success "Setup completed successfully!"
    log_info "To get started:"
    echo "  1. Activate the virtual environment: source venv/bin/activate"
    echo "  2. Set environment variables: source .env"
    echo "  3. Run tests: python -m pytest tests/"
    echo "  4. Start developing!"
}

# Usage function
usage() {
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  setup        Full setup (default)"
    echo "  deps         Install system dependencies only"
    echo "  python       Setup Python environment only"
    echo "  build        Build MLIR components only"
    echo "  install      Install Python package only"
    echo "  test         Run tests"
    echo "  dev          Setup development tools"
    echo "  docs         Generate documentation"
    echo "  verify       Verify installation"
    echo "  clean        Clean build artifacts"
    echo
    echo "Environment Variables:"
    echo "  PYTHON_VERSION    Python version to use (default: 3.11)"
    echo "  LLVM_VERSION      LLVM version to use (default: 17)"
    echo
    echo "Examples:"
    echo "  $0 setup"
    echo "  PYTHON_VERSION=3.10 $0 setup"
    echo "  $0 build"
    echo "  $0 test"
}

# Handle commands
case "${1:-setup}" in
    setup)
        setup
        ;;
    deps)
        detect_os
        install_system_deps
        ;;
    python)
        setup_python
        ;;
    build)
        build_mlir
        ;;
    install)
        install_package
        ;;
    test)
        run_tests
        ;;
    dev)
        setup_dev_tools
        ;;
    docs)
        generate_docs
        ;;
    verify)
        verify_installation
        ;;
    clean)
        clean
        ;;
    -h|--help)
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac