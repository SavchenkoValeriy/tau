//===- AliasAnalysis.h - Alias analysis -------------------------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  TBD
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/Value.h>

namespace mlir {
class AnalysisManager;
} // end namespace mlir

namespace tau::core {

class AliasAnalysis {
public:
  AliasAnalysis(mlir::Operation *Function, mlir::AnalysisManager &AM);

  using Aliases = llvm::SmallDenseSet<mlir::Value>;
  [[nodiscard]] const Aliases &getAliases(mlir::Value Value) const {
    if (const auto It = Sets.find(Value); It != Sets.end()) {
      return It->getSecond();
    }
    return Empty;
  }

private:
  llvm::DenseMap<mlir::Value, Aliases> Sets;
  static Aliases Empty;
};

} // end namespace tau::core
