# ADR-0001: MLIR Dialect Design for Photonic Operations

**Status:** Accepted  
**Date:** 2024-01-15  
**Deciders:** Core Architecture Team  
**Technical Story:** Design of custom MLIR dialect for photonic computing primitives

## Context

The photonic-mlir-synth-bridge requires a way to represent photonic operations in MLIR IR. Standard MLIR dialects (arithmetic, tensor, etc.) do not capture the unique characteristics of photonic operations like phase relationships, wavelength dependencies, and optical power constraints.

## Decision Drivers

* Need to represent complex photonic operations (MZI, phase shifters, wavelength routing)
* Support for photonic-specific optimizations (thermal management, wavelength allocation)
* Integration with existing MLIR infrastructure and optimization passes
* Ability to model both quantum and classical photonic computations
* Support for multi-wavelength operations and wavelength division multiplexing (WDM)
* Hardware-specific constraints (fabrication tolerances, power limits)

## Considered Options

1. **Custom Photonic Dialect**: Create a new MLIR dialect specifically for photonic operations
2. **Extended Standard Dialects**: Extend existing MLIR dialects with photonic attributes
3. **External Representation**: Use external IR format and convert to/from MLIR

## Decision Outcome

Chosen option: "Custom Photonic Dialect", because it provides the most flexibility for photonic-specific optimizations while maintaining clean separation from standard operations.

### Positive Consequences

* Full control over operation semantics and optimization opportunities
* Clean abstraction for photonic concepts (wavelengths, phases, optical power)
* Integration with MLIR optimization infrastructure
* Extensible design for future photonic computing advances
* Support for hardware-specific constraints and optimizations

### Negative Consequences

* Additional maintenance burden for dialect implementation
* Learning curve for developers unfamiliar with photonic computing
* Need to implement custom verification and analysis passes

## Pros and Cons of the Options

### Custom Photonic Dialect

* Good, because provides native representation of photonic concepts
* Good, because enables photonic-specific optimizations
* Good, because maintains clean separation of concerns
* Good, because integrates well with MLIR infrastructure
* Bad, because requires significant implementation effort
* Bad, because adds complexity to the system

### Extended Standard Dialects

* Good, because reuses existing MLIR infrastructure
* Good, because lower implementation overhead
* Bad, because photonic concepts don't map well to standard operations
* Bad, because limits optimization opportunities
* Bad, because creates semantic confusion

### External Representation

* Good, because could reuse existing photonic simulation formats
* Bad, because loses MLIR optimization benefits
* Bad, because requires constant conversion overhead
* Bad, because poor integration with compilation pipeline

## Implementation Details

### Core Operations

```mlir
// Basic photonic operations
photonic.mzi %input1, %input2 {theta = 1.57 : f64, phi = 0.0 : f64} 
  : (complex<f32>, complex<f32>) -> (complex<f32>, complex<f32>)

photonic.phase_shift %input {phase = 0.785 : f64} 
  : complex<f32> -> complex<f32>

photonic.photodetector %input {responsivity = 1.0 : f64} 
  : complex<f32> -> f32
```

### Type System

* `complex<fN>` - Complex amplitude representation
* `!photonic.wavelength<f64>` - Wavelength specification
* `!photonic.power<f64>` - Optical power with units
* `!photonic.channel<N>` - Multi-wavelength channel

### Attributes

* Wavelength specifications for WDM operations
* Power constraints and thermal characteristics
* Fabrication tolerances and process variations
* Hardware-specific parameters (Q-factors, coupling ratios)

## Links

* [MLIR Dialect Developer Guide](https://mlir.llvm.org/docs/DefiningDialects/)
* [Photonic Computing Fundamentals](docs/photonic-computing-primer.md)