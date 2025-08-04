#include "PhotonicMLIR/Dialect/PhotonicDialect.h"
#include "PhotonicMLIR/Conversion/PhotonicToHLS.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Translation.h"

using namespace mlir;

namespace {

// Register HLS translation
void registerPhotonic ToHLSTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-hls", "translate photonic MLIR to HLS",
      [](Operation *op, raw_ostream &output) {
        if (!isa<ModuleOp>(op))
          return failure();
        
        photonic::HLSConfig config;
        std::string hlsCode = photonic::generateHLSCode(cast<ModuleOp>(op), config);
        output << hlsCode;
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<photonic::PhotonicDialect>();
        registry.insert<func::FuncDialect>();
      });
}

} // namespace

int main(int argc, char **argv) {
  registerPhotonic ToHLSTranslation();

  return failed(mlirTranslateMain(argc, argv, "Photonic MLIR translation driver\n"));
}