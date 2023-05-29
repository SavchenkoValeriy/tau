//===- EscapeAnalysis.h - Escape analysis -----------------------*- C++ -*-===//
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

#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/Value.h>

namespace mlir {

class AnalysisManager;
class Operation;

} // end namespace mlir

namespace tau::core {

class EscapeAnalysis {
public:
  EscapeAnalysis(mlir::Operation *Function, mlir::AnalysisManager &AM);

  using EscapeSet = llvm::SmallDenseSet<mlir::Value>;
  [[nodiscard]] const EscapeSet &getEscapeSet() const { return Escapes; }

private:
  EscapeSet Escapes;
};

} // end namespace tau::core
