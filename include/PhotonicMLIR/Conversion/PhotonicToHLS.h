#ifndef PHOTONIC_MLIR_CONVERSION_PHOTONICTOHLS_H
#define PHOTONIC_MLIR_CONVERSION_PHOTONICTOHLS_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
namespace photonic {

/// Configuration for HLS code generation
struct HLSConfig {
    std::string targetPDK = "AIM_Photonics_PDK";
    std::string processNode = "45nm_SOI";
    float powerBudget = 100.0f; // mW
    int wavelengthChannels = 4;
    bool enableThermalOptimization = true;
    bool enableNoiseReduction = true;
};

/// Creates a pass to convert Photonic dialect operations to HLS code
std::unique_ptr<OperationPass<ModuleOp>> createConvertPhotonicToHLSPass(
    const HLSConfig &config = HLSConfig{});

/// Generates HLS code from a photonic module
std::string generateHLSCode(ModuleOp module, const HLSConfig &config);

} // namespace photonic
} // namespace mlir

#endif // PHOTONIC_MLIR_CONVERSION_PHOTONICTOHLS_H