//===- LazyValue.h - Lazy initialized value ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  TBD
//
//===----------------------------------------------------------------------===//

#pragma once

#include <functional>
#include <optional>

namespace tau::support {

template <typename T> class LazyValue {
private:
  std::optional<T> Value;
  std::function<T()> Generator;

public:
  /*implicit*/ LazyValue(const T &V) : Value(V) {}
  /*implicit*/ LazyValue(std::function<T()> Gen) : Generator(std::move(Gen)) {}

  operator T() {
    if (!Value && Generator) {
      Value = Generator();
    }
    return *Value;
  }
};

} // end namespace tau::support
