//===- MutualExclusionAnalysis.h - Mutual Exclusion Analysis ----*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the MutualExclusionAnalysis class, which determines
/// whether two basic blocks or operations are mutually exclusive in the
/// control flow graph.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SmallVector.h>

namespace mlir {
class AnalysisManager;
class Block;
class Operation;
} // namespace mlir

namespace tau::core {

class TopoOrderBlockEnumerator;

/// Analyzes the CFG to determine mutual exclusivity of blocks and operations.
class MutualExclusionAnalysis {
public:
  MutualExclusionAnalysis(mlir::Operation *Op, mlir::AnalysisManager &AM);

  /// Determines if two blocks are mutually exclusive in the CFG.
  ///
  /// @param A -- The first block to check
  /// @param B -- The second block to check
  ///
  /// Two blocks are considered mutually exclusive if there is no path
  /// in the CFG that includes both blocks.
  bool areMutuallyExclusive(const mlir::Block *A, const mlir::Block *B) const;

  /// Determines if two operations are mutually exclusive in the CFG.
  ///
  /// @param A -- The first operation to check
  /// @param B -- The second operation to check
  ///
  /// Two operations are considered mutually exclusive if their parent blocks
  /// are mutually exclusive in the CFG.
  bool areMutuallyExclusive(mlir::Operation *A, mlir::Operation *B) const;

private:
  const TopoOrderBlockEnumerator &TopoOrder;
  llvm::SmallVector<llvm::BitVector, 20> ReachabilityMatrix;

  void computeReachabilityMatrix(mlir::Operation *Op);
};

} // namespace tau::core
