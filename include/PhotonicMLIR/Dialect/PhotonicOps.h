#ifndef PHOTONIC_MLIR_DIALECT_PHOTONIC_OPS_H
#define PHOTONIC_MLIR_DIALECT_PHOTONIC_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "PhotonicMLIR/Dialect/PhotonicOps.h.inc"

#endif // PHOTONIC_MLIR_DIALECT_PHOTONIC_OPS_H