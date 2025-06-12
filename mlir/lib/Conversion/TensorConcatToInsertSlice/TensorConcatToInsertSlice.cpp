//===- TensorConcatToInsertSlice.cpp - Tensor Concat to Linalg Insert Slice -----------===//
//
// Oscar wrote this :)
// This file implements a pass to convert tensor.concat function to linalg insert
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TensorConcatToInsertSlice/TensorConcatToInsertSlice.h"
#include "mlir/Conversion/TensorConcatToInsertSlice/TensorConcatToInsertSlice.h.inc"


#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::tensor;
using namespace mlir::func;

namespace {

/// Pattern to lower tensor.concat into a sequence of tensor.insert_slice
struct TensorConcatToInsertSlice : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    IntegerAttr dimAttr = op->getAttrOfType<IntegerAttr>("dim");
    if (!dimAttr)
      return failure();
    int64_t dim = dimAttr.getInt();

    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    unsigned rank = resultType.getRank();

    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(rank);

    Value result = init;
    int64_t currentOffset = 0;

    for (Value input : op.getInputs()) {
      auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
      if (!inputType)
        return failure();

      sizes.clear();
      for (unsigned i = 0; i < rank; ++i) {
        if (inputType.isDynamicDim(i)) {
          Value dimValue = rewriter.create<tensor::DimOp>(loc, input, i);
          sizes.push_back(dimValue);
        } else {
          sizes.push_back(rewriter.getIndexAttr(inputType.getDimSize(i)));
        }
      }

      offsets.assign(rank, rewriter.getIndexAttr(0));
      offsets[dim] = rewriter.getIndexAttr(currentOffset);

      result = rewriter.create<tensor::InsertSliceOp>(loc, input, result, offsets, sizes, strides);
      currentOffset += inputType.getDimSize(dim);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Pass that applies the rewrite pattern
struct LowerTensorConcatPass
    : public PassWrapper<LowerTensorConcatPass, OperationPass<func::FuncOp>> {

  StringRef getArgument() const override { return "lower-tensor-concat"; }
  StringRef getDescription() const override {
    return "Lower tensor.concat ops into tensor.insert_slice chains for bufferization.";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TensorConcatToInsertSlice>(ctx);

    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace mlir {
namespace {
#define GEN_PASS_DEF_LOWERTENSORCONCATPASS
#include "mlir/Conversion/TensorConcatToInsertSlice/TensorConcatToInsertSlice.h.inc"

struct LowerTensorConcatPass : public impl::LowerTensorConcatPassBase<LowerTensorConcatPass> {
  void runOnOperation() override {
    // Your existing pass implementation
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTensorConcatPass() {
  return std::make_unique<LowerTensorConcatPass>();
}


void registerLowerTensorConcatPass() {
  PassRegistration<LowerTensorConcatPass>();
  }
} // namespace mlir
