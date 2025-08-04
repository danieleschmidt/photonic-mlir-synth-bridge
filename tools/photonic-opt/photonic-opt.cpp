#include "PhotonicMLIR/Dialect/PhotonicDialect.h"
#include "PhotonicMLIR/Transforms/Passes.h"
#include "PhotonicMLIR/Conversion/TorchToPhotonic.h"
#include "PhotonicMLIR/Conversion/PhotonicToHLS.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<photonic::PhotonicDialect>();

  // Register photonic passes
  photonic::registerPhotonic passes();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "Photonic MLIR optimizer driver\n", registry));
}