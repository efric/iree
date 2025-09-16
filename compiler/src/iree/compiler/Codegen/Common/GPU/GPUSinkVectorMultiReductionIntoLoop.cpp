// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"


#define DEBUG_TYPE "iree-llvmgpu-reduction-sink"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUSINKVECTORMULTIREDUCTIONINTOLOOPPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct SinkCandidate {
  vector::MultiDimReductionOp reduction;
  scf::ForOp forOp;
  unsigned loopResultIndex;
};

static void collectSinkCandidates(FunctionOpInterface funcOp,
                                  SmallVectorImpl<SinkCandidate> &out) {
  funcOp.walk([&](vector::MultiDimReductionOp redOp) {
    if (redOp.getKind() != vector::CombiningKind::ADD) {
      return;
    }

    auto srcResult = dyn_cast<OpResult>(redOp.getSource());
    if (!srcResult) {
      return;
    }

    auto forOp = dyn_cast_or_null<scf::ForOp>(srcResult.getOwner());
    if (!forOp) {
      return;
    }
    
    // Source must be a loop result with single use (the reduction).
    unsigned resIdx = srcResult.getResultNumber();
    Value loopRes = forOp.getResult(resIdx);
    if (!loopRes.hasOneUse() || *loopRes.user_begin() != redOp.getOperation()) {
      return;
    }

    out.push_back({redOp, forOp, resIdx});
  });
}

static LogicalResult attemptSinkIntoLoop(IRRewriter &rewriter,
                                         const SinkCandidate &cand) {
  auto [redOp, forOp, resIdx] = cand;
  
  VectorType srcVecType = redOp.getSource().getType();
  //tbd
  auto reducedType = dyn_cast<VectorType>(redOp.getResult().getType());
  if (!reducedType) {
    return failure();
  }
  srcVecType.dump();
  

  return success();
}


struct GPUSinkVectorMultiReductionIntoLoopPass final
    : impl::GPUSinkVectorMultiReductionIntoLoopPassBase<
          GPUSinkVectorMultiReductionIntoLoopPass> {
  using GPUSinkVectorMultiReductionIntoLoopPassBase::
      GPUSinkVectorMultiReductionIntoLoopPassBase;
  void runOnOperation() override;
};
} // namespace

void GPUSinkVectorMultiReductionIntoLoopPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(funcOp);

  SmallVector<SinkCandidate> candidates;
  collectSinkCandidates(funcOp, candidates);
  for (const SinkCandidate &cand : candidates) {
    if (failed(attemptSinkIntoLoop(rewriter, cand))) {
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler