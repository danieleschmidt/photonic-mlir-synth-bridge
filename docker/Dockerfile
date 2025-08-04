# Photonic MLIR Docker Image
# Multi-stage build for production deployment

# Build stage - includes LLVM/MLIR compilation
FROM ubuntu:22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LLVM_VERSION=17
ENV PYTHON_VERSION=3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    wget \
    curl \
    clang-${LLVM_VERSION} \
    llvm-${LLVM_VERSION}-dev \
    libmlir-${LLVM_VERSION}-dev \
    mlir-${LLVM_VERSION}-tools \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy source code
WORKDIR /src
COPY . .

# Build MLIR components
RUN mkdir -p build && cd build && \
    cmake -G Ninja .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DMLIR_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/mlir \
        -DLLVM_EXTERNAL_LIT=/usr/bin/llvm-lit-${LLVM_VERSION} && \
    ninja

# Build Python package
RUN pip3 install -e .

# Runtime stage - minimal image for production
FROM ubuntu:22.04 as runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
ENV LLVM_VERSION=17

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    libmlir-${LLVM_VERSION} \
    llvm-${LLVM_VERSION}-runtime \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Copy built artifacts from builder stage
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages /usr/local/lib/python${PYTHON_VERSION}/dist-packages
COPY --from=builder /src/build/bin /usr/local/bin
COPY --from=builder /src/build/lib /usr/local/lib

# Create non-root user for security
RUN groupadd -r photonic && useradd -r -g photonic -s /bin/bash photonic
RUN mkdir -p /app /data /logs && chown -R photonic:photonic /app /data /logs

# Set working directory
WORKDIR /app
USER photonic

# Environment variables
ENV PYTHONPATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages:${PYTHONPATH}"
ENV PATH="/usr/local/bin:${PATH}"
ENV PHOTONIC_CACHE_DIR="/data/cache"
ENV PHOTONIC_LOG_DIR="/logs"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import photonic_mlir; print('Photonic MLIR OK')" || exit 1

# Default command
CMD ["python", "-c", "import photonic_mlir; print('Photonic MLIR container ready')"]

# Labels for metadata
LABEL maintainer="Photonic MLIR Team <team@photonic-mlir.org>"
LABEL version="0.1.0"
LABEL description="Silicon Photonic AI Accelerator Compiler"
LABEL org.opencontainers.image.source="https://github.com/danieleschmidt/photonic-mlir-synth-bridge"