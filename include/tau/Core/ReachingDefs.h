//===- ReachingDefs.h - Reaching definitions --------------------*- C++ -*-===//
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

#include <memory>

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>

namespace mlir {

class AnalysisManager;
class Operation;

} // end namespace mlir

namespace tau::core {

class ReachingDefs {
public:
  ReachingDefs(mlir::Operation *Function, mlir::AnalysisManager &AM);

  using Definitions = llvm::SmallVector<mlir::Value, 8>;
  Definitions getDefinitions(mlir::Operation &At, mlir::Value For) const;

  ReachingDefs(const ReachingDefs &) = delete;
  ReachingDefs &operator=(const ReachingDefs &) = delete;

  ReachingDefs(ReachingDefs &&);
  ReachingDefs &operator=(ReachingDefs &&);

  ~ReachingDefs();

private:
  class Implementation;
  std::unique_ptr<Implementation> PImpl;
};
} // end namespace tau::core

#pragma once
