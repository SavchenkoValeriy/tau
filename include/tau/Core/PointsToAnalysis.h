//===- PointsToAnalysis.h - Points-to analysis ------------------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  TBD
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>

namespace mlir {

class AnalysisManager;
class Operation;

} // end namespace mlir

namespace tau::core {

class PointsToAnalysis {
public:
  PointsToAnalysis(mlir::Operation *Function, mlir::AnalysisManager &AM);

  using PointsToSet = llvm::SmallVector<mlir::Value, 4>;
  [[nodiscard]] const PointsToSet &getPointsToSet(mlir::Value Pointer) const {
    if (const auto It = Sets.find(Pointer); It != Sets.end()) {
      return It->getSecond();
    }
    return Empty;
  }

private:
  llvm::DenseMap<mlir::Value, PointsToSet> Sets;
  static PointsToSet Empty;
};

} // end namespace tau::core

#pragma once
