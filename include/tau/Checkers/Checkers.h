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

#include <memory>

namespace mlir {
class Pass;
} // end namespace mlir

namespace tau {
namespace chx {

std::unique_ptr<mlir::Pass> createUseOfUninitChecker();

void registerUseOfUninitChecker();
} // end namespace chx
} // end namespace tau
