//===- FlowSensitive.h - Flow-sensitive analysis ----------------*- C++ -*-===//
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

#include "tau/Core/Events.h"

#include <llvm/ADT/ArrayRef.h>
#include <memory>

namespace mlir {
class AnalysisManager;
class Operation;
} // end namespace mlir

namespace tau::core {

class FlowSensitiveAnalysis {
public:
  FlowSensitiveAnalysis(mlir::Operation *TopLevelOp, mlir::AnalysisManager &AM);

  struct Issue {
    LinearChainOfEvents Events;
    bool Guaranteed;
  };

  EventHierarchy &getEventHierarchy();
  llvm::ArrayRef<Issue> getFoundIssues();

  FlowSensitiveAnalysis(const FlowSensitiveAnalysis &) = delete;
  FlowSensitiveAnalysis &operator=(const FlowSensitiveAnalysis &) = delete;

  FlowSensitiveAnalysis(FlowSensitiveAnalysis &&);
  FlowSensitiveAnalysis &operator=(FlowSensitiveAnalysis &&);

  ~FlowSensitiveAnalysis();

private:
  class Implementation;
  std::unique_ptr<Implementation> PImpl;
};

} // end namespace tau::core
