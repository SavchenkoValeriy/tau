//===- Checkers.h - Full registry of checkers -------------------*- C++ -*-===//
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

#include "tau/AIR/AirAttrs.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace tau {
namespace chx {

template <class CheckerT, class State>
class Checker
    : public mlir::PassWrapper<CheckerT, mlir::OperationPass<mlir::FuncOp>> {
public:
  /// Mark the given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param What -- operand to change the state
  /// @param To -- target state to change the operand to
  ///
  /// Since this version doesn't specify which state the value should be in
  /// before, it will change any value to the @p To state only if it wasn't in
  /// any state.
  void markChange(mlir::Operation *Where, mlir::Value What, State To) {
    unsigned WhatIdx = findOperand(Where, What);
    markChangeImpl(Where, WhatIdx, To);
  }

  /// Mark the result of given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param To -- target state to change the operand to
  ///
  /// Since this version doesn't specify which state the value should be in
  /// before, it will change any value to the @p To state only if it wasn't in
  /// any state.
  void markResultChange(mlir::Operation *Where, State To) {
    unsigned ResultIdx = getResultIdx(Where);
    markChangeImpl(Where, ResultIdx, To);
  }

  /// Mark the given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param What -- operand to change the state
  /// @param From -- we only change the state of the value if it
  /// is in this state
  /// @param To -- target state to change the operand to
  void markChange(mlir::Operation *Where, mlir::Value What, State From,
                  State To) {
    unsigned WhatIdx = findOperand(Where, What);
    markChangeImpl(Where, WhatIdx, From, To);
  }

  /// Mark the result of given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param From -- we only change the state of the value if it
  /// is in this state
  /// @param To -- target state to change the operand to
  void markResultChange(mlir::Operation *Where, State From, State To) {
    unsigned WhatIdx = getResultIdx(Where);
    markChangeImpl(Where, WhatIdx, From, To);
  }

private:
  template <class... Args>
  void markChangeImpl(mlir::Operation *Where, Args &&...Rest) {
    // TODO: figure out how to have miltiple instances of State attributes
    Where->setAttr(this->getArgument(),
                   air::StateChangeAttr::get(Where->getContext(),
                                             this->getArgument(),
                                             std::forward<Args>(Rest)...));
  }

  static unsigned findOperand(mlir::Operation *Where, mlir::Value What) {
    auto AllOperands = Where->getOperands();
    auto It = llvm::find(AllOperands, What);
    if (It != AllOperands.end())
      return It - AllOperands.begin();

    // All operations in Air have exactly one result
    assert(What == Where->getOpResult(0));
    return getResultIdx(Where);
  }

  static unsigned getResultIdx(mlir::Operation *Of) {
    assert(Of->getNumResults() == 1 &&
           "The given operation should have a result to begin with");
    return Of->getNumOperands();
  }
};

std::unique_ptr<mlir::Pass> createUseOfUninitChecker();

void registerUseOfUninitChecker();
} // end namespace chx
} // end namespace tau
