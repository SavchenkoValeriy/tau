//===- CheckerRegistry.h - Registry for all the checkers --------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  This file defines basic interface for concrete checker registration and
//  partial checker enablement via CLI flags.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tau/Core/Checker.h"

#include <llvm/ADT/StringRef.h>

#include <functional>
#include <memory>

namespace llvm::cl {
class OptionCategory;
} // end namespace llvm::cl

namespace mlir {
class PassManager;
} // end namespace mlir

namespace tau::core {

using CheckerAllocatorFunction =
    std::function<std::unique_ptr<core::AbstractChecker>()>;

void registerChecker(const CheckerAllocatorFunction &Constructor);
core::AbstractChecker &findChecker(llvm::StringRef Argument);

template <Checker T> struct CheckerRegistration {
  CheckerRegistration(const CheckerAllocatorFunction &Constructor) {
    registerChecker(Constructor);
  }
  CheckerRegistration()
      : CheckerRegistration([] { return std::make_unique<T>(); }) {}
};

class CheckerCLParser {
public:
  CheckerCLParser(llvm::cl::OptionCategory &CategoryForCheckerOptions);

  CheckerCLParser(const CheckerCLParser &) = delete;
  CheckerCLParser &operator=(const CheckerCLParser &) = delete;

  CheckerCLParser(CheckerCLParser &&);
  CheckerCLParser &operator=(CheckerCLParser &&);

  ~CheckerCLParser();

  /// Add all enabled checkers to the given pass manager.
  void addEnabledCheckers(mlir::PassManager &PM);

private:
  class Implementation;
  std::unique_ptr<Implementation> PImpl;
};

} // end namespace tau::core
