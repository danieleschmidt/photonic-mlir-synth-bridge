#ifndef PHOTONIC_MLIR_CONVERSION_TORCHTOPHOTONIC_H
#define PHOTONIC_MLIR_CONVERSION_TORCHTOPHOTONIC_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace photonic {

/// Creates a pass to convert Torch dialect operations to Photonic dialect
std::unique_ptr<OperationPass<ModuleOp>> createConvertTorchToPhotonicPass();

/// Populates patterns for converting Torch operations to Photonic operations
void populateTorchToPhotonicConversionPatterns(RewritePatternSet &patterns,
                                               TypeConverter &typeConverter);

} // namespace photonic
} // namespace mlir

#endif // PHOTONIC_MLIR_CONVERSION_TORCHTOPHOTONIC_H