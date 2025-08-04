#ifndef PHOTONIC_MLIR_TRANSFORMS_PASSES_H
#define PHOTONIC_MLIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace photonic {

//===----------------------------------------------------------------------===//
// Photonic Optimization Passes
//===----------------------------------------------------------------------===//

/// Creates a pass for wavelength allocation optimization
std::unique_ptr<Pass> createWavelengthAllocationPass();

/// Creates a pass for thermal-aware placement optimization
std::unique_ptr<Pass> createThermalOptimizationPass();

/// Creates a pass for phase quantization
std::unique_ptr<Pass> createPhaseQuantizationPass(int quantizationBits = 8);

/// Creates a pass for power gating optimization
std::unique_ptr<Pass> createPowerGatingPass();

/// Creates a pass for coherent noise reduction
std::unique_ptr<Pass> createCoherentNoiseReductionPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "PhotonicMLIR/Transforms/Passes.h.inc"

} // namespace photonic
} // namespace mlir

#endif // PHOTONIC_MLIR_TRANSFORMS_PASSES_H