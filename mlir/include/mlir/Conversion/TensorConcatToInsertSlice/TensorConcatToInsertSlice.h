#ifndef MLIR_CONVERSION_TENSORCONCATTOINSERTSLICE_H
#define MLIR_CONVERSION_TENSORCONCATTOINSERTSLICE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_LOWERTENSORCONCATPASS
#include "mlir/Conversion/TensorConcatToInsertSlice/TensorConcatToInsertSlice.h.inc"

std::unique_ptr<Pass> createLowerTensorConcatPass();
} // namespace mlir

#endif
