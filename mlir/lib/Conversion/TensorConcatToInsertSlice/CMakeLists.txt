set(LLVM_TARGET_DEFINITIONS TensorConcatToInsertSlice.td)
mlir_tablegen(TensorConcatToInsertSlice.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRTensorConcatToInsertSliceIncGen)

add_mlir_conversion_library(MLIRTensorConcatToInsertSlice
  TensorConcatToInsertSlice.cpp
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  
  DEPENDS
  MLIRTensorConcatToInsertSliceIncGen
)
