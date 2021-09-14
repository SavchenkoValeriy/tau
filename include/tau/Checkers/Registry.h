//===- Registry.h - Registry for all the checkers ---------------*- C++ -*-===//
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

#include "tau/Checkers/Checkers.h"

#include <llvm/ADT/StringRef.h>

#include <functional>
#include <memory>

namespace tau::chx {

using CheckerAllocatorFunction = std::function<std::unique_ptr<Checker>()>;

void registerChecker(const CheckerAllocatorFunction &Constructor);

template <typename ConcreteChecker> struct CheckerRegistration {
  CheckerRegistration(const CheckerAllocatorFunction &Constructor) {
    registerChecker(Constructor);
  }
  CheckerRegistration()
      : CheckerRegistration(
            [] { return std::make_unique<ConcreteChecker>(); }) {}
};

} // end namespace tau::chx
