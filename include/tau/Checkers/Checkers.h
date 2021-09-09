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

#include "tau/Core/State.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace tau {
namespace chx {

template <class CheckerT>
class Checker
    : public mlir::PassWrapper<CheckerT, mlir::OperationPass<mlir::FuncOp>> {
public:
  template <unsigned X, unsigned Y>
  void mark(mlir::Operation *ToMark, core::State<X, Y> WithWhat) {
    // TODO: Change to Air own attributes
    ToMark->setAttr(this->getArgument(),
                    mlir::BoolAttr::get(ToMark->getContext(), true));
  }
};

std::unique_ptr<mlir::Pass> createUseOfUninitChecker();

void registerUseOfUninitChecker();
} // end namespace chx
} // end namespace tau
