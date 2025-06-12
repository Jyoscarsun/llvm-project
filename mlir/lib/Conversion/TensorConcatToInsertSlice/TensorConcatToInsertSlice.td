include "mlir/Pass/PassBase.td"

def LowerTensorConcatPass : Pass<"lower-tensor-concat"> {
  let summary = "Lower tensor.concat to tensor.insert_slice operations";
  let description = [{
    Converts tensor.concat operations into sequences of tensor.insert_slice operations.
  }];
  let dependentDialects = ["tensor::TensorDialect"];
}
