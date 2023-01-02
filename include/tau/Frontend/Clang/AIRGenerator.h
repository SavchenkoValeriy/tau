//===- AIRGenerator.h - Generation of AIR -----------------------*- C++ -*-===//
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

#include "tau/Frontend/Clang/OwningModuleRef.h"

namespace mlir {
class MLIRContext;
} // end namespace mlir

namespace clang {
class ASTContext;
} // end namespace clang

namespace tau {
namespace frontend {

class AIRGenerator {
public:
  static OwningModuleRef generate(mlir::MLIRContext &, clang::ASTContext &);
};
} // end namespace frontend
} // end namespace tau
