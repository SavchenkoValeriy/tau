//===- EventForest.h - Storage for event forests ----------------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  Event forest is a data-structure that allows us to represent many connected
//  events and have a unified ownership of all allocated events.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/Support/Allocator.h>

#include <concepts>

namespace tau::core {

template <class T>
concept Event = requires(T &Candidate) {
  { Candidate.Parent } -> std::same_as<const T *&>;
};

template <Event ConcreteEvent> class EventForest {
private:
  // As of now, forests only track the memory and nothing more.
  llvm::SpecificBumpPtrAllocator<ConcreteEvent> Allocator;

public:
  template <class... Args> const ConcreteEvent &addEvent(Args &&...Rest) {
    return *(new (Allocator.Allocate()) ConcreteEvent{Rest...});
  }
};

} // end namespace tau::core
