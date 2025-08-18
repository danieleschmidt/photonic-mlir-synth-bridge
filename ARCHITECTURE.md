# Architecture Documentation

## System Overview

The photonic-mlir-synth-bridge is a comprehensive compiler infrastructure that translates high-level neural network descriptions into optimized silicon photonic circuits. The system follows a multi-layered architecture designed for scalability, extensibility, and production deployment.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend Layer                               │
├─────────────────────────────────────────────────────────────────┤
│ PyTorch Frontend │ TensorFlow Frontend │ JAX Frontend │ ONNX    │
│                  │                     │              │         │
│ pytorch_frontend.py                    │ conversion/  │         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   MLIR Dialect Layer                           │
├─────────────────────────────────────────────────────────────────┤
│ Photonic Dialect │ Operations │ Types │ Attributes │ Interfaces │
│                  │            │       │            │            │
│ PhotonicDialect.td             │ PhotonicOps.td    │           │
│ PhotonicTypes.td               │ PhotonicDialect.h │           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                 Optimization Layer                             │
├─────────────────────────────────────────────────────────────────┤
│ Passes │ Transformations │ Analysis │ Optimizations │ Validation│
│        │                 │          │               │           │
│ WavelengthAllocation     │ ThermalOptimization       │          │
│ NoiseReduction           │ PowerGating               │          │
│ PhaseQuantization        │ CoherentNoiseSuppression  │          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  Backend Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│ HLS Generation │ Simulation │ Hardware Interface │ Deployment   │
│                │            │                    │              │
│ PhotonicToHLS  │ simulation.py                   │ hardware/    │
│ Netlist Gen    │ PhotonicSimulator               │ interfaces   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                Infrastructure Layer                            │
├─────────────────────────────────────────────────────────────────┤
│ Caching │ Monitoring │ Security │ Load Balancing │ Auto-scaling │
│         │            │          │                │              │
│ cache.py│monitoring.py│security.py│load_balancer.py│            │
│ Redis   │Prometheus  │Sandboxing │Adaptive LB     │Kubernetes   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Compiler Core (`compiler.py`)
- **Purpose**: Main compilation orchestration and coordination
- **Key Classes**: `PhotonicCompiler`, `CompilerCore`
- **Interfaces**: Frontend integration, optimization pipeline, backend generation
- **Dependencies**: MLIR bindings, optimization modules, validation

### 2. MLIR Dialect Implementation
- **Location**: `include/PhotonicMLIR/Dialect/`
- **Components**:
  - `PhotonicDialect.td`: Dialect definition and registration
  - `PhotonicOps.td`: Operation definitions (MZI, phase shifters, detectors)
  - `PhotonicTypes.td`: Type system (complex numbers, wavelengths, power)
- **Key Features**: Custom attributes for photonic parameters, verification hooks

### 3. Frontend Integration Layer
- **PyTorch Frontend** (`pytorch_frontend.py`): Graph extraction, IR conversion
- **TensorFlow Support**: Via MLIR TensorFlow dialect conversion
- **ONNX Support**: Import through ONNX-MLIR integration

### 4. Optimization Pipeline
- **Photonic-Specific Passes**:
  - Wavelength allocation and WDM optimization
  - Thermal-aware placement and routing
  - Phase quantization for fabrication constraints
  - Power gating and thermal management
- **Standard Optimizations**: Dead code elimination, constant folding, loop optimization

### 5. Backend Generation
- **HLS Generation**: Synthesizable RTL for foundry PDKs
- **Netlist Export**: SPICE-compatible circuit descriptions
- **Layout Generation**: GDS-II physical layout files
- **Simulation Interface**: Cycle-accurate behavioral models

## Data Flow Architecture

```
Input Model → Frontend Parsing → MLIR IR Generation → Optimization Passes → Backend Generation → Hardware Deployment
     │              │                    │                     │                    │                    │
     ├─PyTorch      ├─Graph Analysis     ├─Photonic Dialect    ├─Wavelength Opt    ├─HLS Code          ├─Chip Programming
     ├─TensorFlow   ├─Type Inference     ├─Operation Lowering  ├─Thermal Analysis  ├─Layout Files      ├─Calibration
     └─ONNX         └─Validation         └─Dialect Conversion  └─Power Optimization└─Simulation Model  └─Testing
```

## Distributed Architecture

### Microservices Design
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Compilation   │    │   Optimization  │    │   Simulation    │
│    Service      │◄──►│    Service      │◄──►│    Service      │
│                 │    │                 │    │                 │
│ - Graph parsing │    │ - Pass pipeline │    │ - Circuit sim   │
│ - IR generation │    │ - Thermal opt   │    │ - Performance   │
│ - Validation    │    │ - Power gating  │    │ - Verification  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────▼─────────────────────────────────┐
│                     Shared Infrastructure                        │
│                                                                   │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│ │  Cache  │ │Monitor  │ │Security │ │Load Bal │ │Metadata │      │
│ │ (Redis) │ │(Prom)   │ │         │ │         │ │  Store  │      │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘      │
└───────────────────────────────────────────────────────────────────┘
```

## Security Architecture

### Multi-Layer Security Model
1. **Input Validation**: Comprehensive sanitization at ingress points
2. **Sandboxing**: Isolated execution environments for untrusted code
3. **Access Control**: Role-based permissions for system components
4. **Audit Logging**: Complete operation traceability
5. **Network Security**: TLS encryption, network segmentation

### Security Boundaries
```
External Input → Input Validation → Sandboxed Execution → Validated Output → External Systems
      │               │                    │                     │               │
   Untrusted       Sanitized           Contained            Verified        Trusted
```

## Scalability Design

### Horizontal Scaling Points
- **Compilation Workers**: Multiple parallel compilation processes
- **Optimization Pipeline**: Distributed pass execution
- **Simulation Cluster**: Parallel circuit simulation instances
- **Cache Layer**: Distributed Redis cluster with sharding

### Vertical Scaling Features
- **Memory Management**: Efficient IR representation, streaming processing
- **CPU Optimization**: Multi-threading, SIMD instructions
- **I/O Optimization**: Asynchronous file operations, compression

## Integration Points

### External Systems
- **Hardware Interfaces**: Direct chip programming, calibration systems
- **Cloud Platforms**: AWS, GCP, Azure integration for scaling
- **Development Tools**: IDE integrations, debugging interfaces
- **Monitoring Systems**: Grafana dashboards, alerting platforms

### API Boundaries
- **REST API**: HTTP endpoints for external tool integration
- **gRPC Services**: High-performance internal communication
- **Python API**: Native library interface for research workflows
- **CLI Tools**: Command-line utilities for batch operations

## Performance Characteristics

### Compilation Performance
- **Small Models** (< 10 layers): < 100ms compilation time
- **Medium Models** (10-50 layers): < 1s compilation time
- **Large Models** (> 50 layers): < 10s compilation time with caching

### Memory Footprint
- **Base System**: < 100MB resident memory
- **Per Compilation**: 10-50MB depending on model complexity
- **Cache Overhead**: Configurable, typically 500MB-2GB

### Throughput Metrics
- **Concurrent Compilations**: 10-100 depending on resource allocation
- **Cache Hit Rate**: > 80% in typical research workflows
- **Network Throughput**: > 1GB/s for distributed deployments

## Quality Assurance

### Testing Strategy
- **Unit Tests**: Component-level verification (> 90% coverage)
- **Integration Tests**: End-to-end compilation pipelines
- **Performance Tests**: Regression testing for compilation speed
- **Security Tests**: Penetration testing, input fuzzing

### Reliability Features
- **Error Recovery**: Graceful degradation on component failures
- **Health Monitoring**: Real-time system health assessment
- **Backup Systems**: Redundant storage, failover mechanisms
- **Version Management**: Backward compatibility, migration paths

## Development Architecture

### Build System
- **CMake**: C++ components, MLIR integration
- **Python setuptools**: Python package management
- **Docker**: Containerized deployment, development environments
- **Kubernetes**: Production orchestration, scaling

### Continuous Integration
- **Testing Pipeline**: Automated test execution on multiple platforms
- **Security Scanning**: Dependency vulnerability assessment
- **Performance Monitoring**: Regression detection, benchmarking
- **Documentation Generation**: Automated API documentation updates