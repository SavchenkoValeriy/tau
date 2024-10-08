//===- CheckerPass.h - Pass running all checkers at once --------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  This file defines an MLIR pass that executes all enabled checkers and
//  modifies AIR code with abstract state attributes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/ArrayRef.h>

#include <memory>

namespace mlir {
class Pass;
} // end namespace mlir

namespace tau::core {

class AbstractChecker;

[[nodiscard]] std::unique_ptr<mlir::Pass>
createCheckerPass(llvm::ArrayRef<AbstractChecker *> CheckersToRun);

} // end namespace tau::core
