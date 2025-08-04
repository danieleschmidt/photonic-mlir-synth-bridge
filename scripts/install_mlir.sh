#!/bin/bash
# Install MLIR/LLVM dependencies for Photonic MLIR

set -e

LLVM_VERSION=${LLVM_VERSION:-17}
BUILD_TYPE=${BUILD_TYPE:-Release}
INSTALL_PREFIX=${INSTALL_PREFIX:-/usr/local}

echo "Installing MLIR/LLVM ${LLVM_VERSION} (${BUILD_TYPE}) to ${INSTALL_PREFIX}"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

install_linux() {
    echo "Installing on Linux..."
    
    # Add LLVM repository
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    echo "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${LLVM_VERSION} main" | \
        sudo tee /etc/apt/sources.list.d/llvm.list
    
    # Update package list
    sudo apt-get update
    
    # Install LLVM/MLIR packages
    sudo apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        git \
        python3-dev \
        llvm-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}-dev \
        llvm-${LLVM_VERSION}-tools \
        libmlir-${LLVM_VERSION}-dev \
        mlir-${LLVM_VERSION}-tools \
        clang-${LLVM_VERSION} \
        lld-${LLVM_VERSION}
    
    # Set up alternatives
    sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-${LLVM_VERSION} 100
    sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100
    sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VERSION} 100
}

install_macos() {
    echo "Installing on macOS..."
    
    # Install via Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first."
        exit 1
    fi
    
    # Install LLVM
    brew install llvm@${LLVM_VERSION}
    
    # Set up PATH
    LLVM_PATH="/opt/homebrew/opt/llvm@${LLVM_VERSION}"
    if [[ -d "$LLVM_PATH" ]]; then
        echo "export PATH=\"$LLVM_PATH/bin:\$PATH\"" >> ~/.bashrc
        echo "export LDFLAGS=\"-L$LLVM_PATH/lib \$LDFLAGS\"" >> ~/.bashrc
        echo "export CPPFLAGS=\"-I$LLVM_PATH/include \$CPPFLAGS\"" >> ~/.bashrc
        echo "Added LLVM paths to ~/.bashrc"
    fi
}

build_from_source() {
    echo "Building LLVM/MLIR from source..."
    
    LLVM_SRC_DIR="llvm-project"
    BUILD_DIR="llvm-build"
    
    # Clone LLVM project if not exists
    if [[ ! -d "$LLVM_SRC_DIR" ]]; then
        git clone --depth 1 --branch "llvmorg-${LLVM_VERSION}.0.0" \
            https://github.com/llvm/llvm-project.git "$LLVM_SRC_DIR"
    fi
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure with CMake
    cmake -G Ninja "../$LLVM_SRC_DIR/llvm" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="Native" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE="$(which python3)"
    
    # Build
    echo "Building LLVM/MLIR (this may take a while)..."
    ninja
    
    # Install
    if [[ "$INSTALL_PREFIX" == "/usr/local" ]]; then
        sudo ninja install
    else
        ninja install
    fi
    
    cd ..
    echo "LLVM/MLIR built and installed to $INSTALL_PREFIX"
}

# Main installation logic
case "$1" in
    "source")
        build_from_source
        ;;
    *)
        if [[ "$OS" == "linux" ]]; then
            install_linux
        elif [[ "$OS" == "macos" ]]; then
            install_macos
        fi
        ;;
esac

# Verify installation
echo "Verifying MLIR installation..."
if command -v mlir-opt &> /dev/null; then
    echo "✓ mlir-opt found: $(which mlir-opt)"
    mlir-opt --version
else
    echo "✗ mlir-opt not found in PATH"
    exit 1
fi

if command -v llvm-config &> /dev/null; then
    echo "✓ llvm-config found: $(which llvm-config)"
    echo "LLVM version: $(llvm-config --version)"
    echo "LLVM prefix: $(llvm-config --prefix)"
else
    echo "✗ llvm-config not found in PATH"
    exit 1
fi

echo "MLIR/LLVM installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Source your shell configuration: source ~/.bashrc"
echo "2. Build Photonic MLIR: make build"
echo "3. Install Python package: make install"