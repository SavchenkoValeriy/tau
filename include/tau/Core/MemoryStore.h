//===- MemoryStore.h - Memory store abstraction -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MemoryStore class, which provides an abstract model
// of program memory state for use in static analysis. The MemoryStore tracks
// relationships between values, including points-to information, field
// accesses, and array elements.
//
// The MemoryStore uses an immutable data structure to represent the memory
// state, allowing efficient updates and comparisons between different states.
// It provides operations to interpret MLIR operations and update the memory
// state accordingly, as well as to join different memory states at control
// flow merge points.
//
// Key components of the MemoryStore include:
// - A model of memory relationships (points-to, field access, array elements)
// - An efficient method for updating the memory state based on operations
// - A mechanism for joining memory states from different control flow paths
// - Methods for querying the memory state to support various analyses
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
