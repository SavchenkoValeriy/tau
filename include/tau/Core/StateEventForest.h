//===- StateEventForest.h - Forest of state-related events ------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
// This file defines the StateEventForest class, which is responsible for
// managing memory and relating state change events.
//
// The StateEventForest maintains a forest of state events, where each tree
// in the forest represents a sequence of state transitions. This structure
// allows the analysis to track the history of state changes and their
// relationships, which is crucial for understanding the evolution of program
// states and detecting potential issues.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tau/AIR/StateID.h"
#include "tau/Core/EventForest.h"

#include <llvm/ADT/StringRef.h>

namespace mlir {
class Operation;
} // end namespace mlir

namespace tau::core {

struct StateKey {
  llvm::StringRef CheckerID;
  air::StateID State;

  bool operator==(const StateKey &Other) const {
    return CheckerID == Other.CheckerID && State == Other.State;
  }
};

struct StateEvent {
  StateKey Key;
  mlir::Operation *Location = nullptr;
  const StateEvent *Parent = nullptr;
};

class StateEventForest : public EventForest<StateEvent> {};

} // end namespace tau::core

namespace std {
template <> struct hash<tau::core::StateKey> {
  size_t operator()(const tau::core::StateKey &Key) const {
    return llvm::hash_combine(llvm::hash_value(Key.State), Key.CheckerID);
  }
};
} // end namespace std
