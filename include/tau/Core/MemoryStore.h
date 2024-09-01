//===- MemoryStore.h - Memory store abstraction -----------------*- C++ -*-===//
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

#include "tau/AIR/AirOps.h"

#include <immer/map.hpp>
#include <immer/set.hpp>

#include <variant>

namespace tau::core {

class MemoryStore {
private:
public:
  MemoryStore();

  MemoryStore(const MemoryStore &);
  MemoryStore &operator=(const MemoryStore &);

  MemoryStore(MemoryStore &&);
  MemoryStore &operator=(MemoryStore &&);

  ~MemoryStore();

  struct MemoryKey;
  [[nodiscard]] MemoryStore interpret(mlir::Operation *Op);
  [[nodiscard]] MemoryStore join(MemoryStore Other);

  using SetOfValues = immer::set<mlir::Value>;
  [[nodiscard]] SetOfValues getDefininingValues(mlir::Value) const;

  bool operator==(const MemoryStore &) const;
  bool operator!=(const MemoryStore &) const;

private:
  using ModelTy = immer::map<MemoryKey, SetOfValues>;
  using CanonicalsTy = immer::map<mlir::Value, SetOfValues>;
  class Builder;

  MemoryStore(ModelTy Model, CanonicalsTy Canonicals);

  ModelTy Model;
  CanonicalsTy Canonicals;
};

} // end namespace tau::core
