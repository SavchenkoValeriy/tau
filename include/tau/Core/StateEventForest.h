//===- StateEventForest.h - Forest of state-related events ------*- C++ -*-===//
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

#include "tau/AIR/StateID.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Allocator.h>

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

class StateEventForest {
private:
  // As of now, forests only track the memory and nothing more.
  llvm::SpecificBumpPtrAllocator<StateEvent> Allocator;

public:
  template <class... Args> const StateEvent &addEvent(Args &&...Rest) {
    return *(new (Allocator.Allocate()) StateEvent{Rest...});
  }
};

} // end namespace tau::core

namespace std {
template <> struct hash<tau::core::StateKey> {
  size_t operator()(const tau::core::StateKey &Key) const {
    return llvm::hash_combine(llvm::hash_value(Key.State), Key.CheckerID);
  }
};
} // end namespace std
