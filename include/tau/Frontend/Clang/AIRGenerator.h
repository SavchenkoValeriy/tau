//===- AIRGenerator.h - Generation of AIR -----------------------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
// This file declares the AIRGenerator class, which is responsible for
// translating Clang AST into AIR.
//
// The AIRGenerator serves as the entry point for the translation process,
// coordinating the conversion of an entire translation unit. It manages the
// creation of MLIR modules, functions, and other top-level constructs that
// correspond to the input C/C++ code.
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
