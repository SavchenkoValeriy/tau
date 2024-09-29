//===- Events.h - Events composing the found issues -------------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  Defines common types for Tau events.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tau/AIR/StateID.h"
#include <initializer_list>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/PointerIntPair.h>
#include <llvm/ADT/PointerUnion.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/TrailingObjects.h>
#include <memory>

namespace mlir {
class Operation;
} // end namespace mlir

namespace tau::core {

class DataFlowEvent;
class StateEvent;
class EventHierarchy;
using ParentEventPointer =
    llvm::PointerUnion<const DataFlowEvent *, const StateEvent *>;

namespace detail {

constexpr size_t BITS_FOR_PARENTS = 3;
constexpr size_t MAX_NUMBER_OF_PARENTS =
    (static_cast<size_t>(1) << BITS_FOR_PARENTS) - 1;

struct CommonBase {
  llvm::PointerIntPair<mlir::Operation *, BITS_FOR_PARENTS>
      LocationAndNumberOfParents;

  CommonBase(mlir::Operation *Location, size_t NumberOfParents)
      : LocationAndNumberOfParents(Location, NumberOfParents) {
    assert(NumberOfParents < MAX_NUMBER_OF_PARENTS &&
           "Can't encode that many parent events!");
  }

  size_t getNumberOfParents() const {
    return LocationAndNumberOfParents.getInt();
  }

  mlir::Operation *getLocation() const {
    return LocationAndNumberOfParents.getPointer();
  }
};

template <class Derived>
class BaseEvent : public CommonBase,
                  public llvm::TrailingObjects<Derived, ParentEventPointer> {
  using TrailingObjects = llvm::TrailingObjects<Derived, ParentEventPointer>;

  friend TrailingObjects;
  friend EventHierarchy;

  static size_t sizeToAlloc(size_t NumberOfParents) {
    return Derived::template totalSizeToAlloc<ParentEventPointer>(
        NumberOfParents);
  }

protected:
  using CommonBase::CommonBase;

  size_t numTrailingObjects(
      TrailingObjects::template OverloadToken<ParentEventPointer>) const {
    return getNumberOfParents();
  }

  Derived *getAsDerived() { return static_cast<Derived *>(this); }
  const Derived *getAsDerived() const {
    return static_cast<const Derived *>(this);
  }
  void initTrailingObjects(std::initializer_list<ParentEventPointer> Pointers) {
    std::uninitialized_copy(
        Pointers.begin(), Pointers.end(),
        getAsDerived()->template getTrailingObjects<ParentEventPointer>());
  }

public:
  llvm::ArrayRef<const ParentEventPointer> getParents() const {
    return {getAsDerived()->template getTrailingObjects<ParentEventPointer>(),
            getNumberOfParents()};
  }
};

} // end namespace detail

class DataFlowEvent final : public detail::BaseEvent<DataFlowEvent> {
public:
  DataFlowEvent(mlir::Operation *Location,
                std::initializer_list<ParentEventPointer> DependsOn)
      : detail::BaseEvent<DataFlowEvent>{Location, DependsOn.size()} {
    initTrailingObjects(DependsOn);
  }
};

struct StateKey {
  llvm::StringRef CheckerID;
  air::StateID State;

  bool operator==(const StateKey &Other) const {
    return CheckerID == Other.CheckerID && State == Other.State;
  }
};

class StateEvent final : public detail::BaseEvent<StateEvent> {
public:
  StateEvent(StateKey Key, mlir::Operation *Location,
             std::initializer_list<ParentEventPointer> DependsOn)
      : detail::BaseEvent<StateEvent>{Location, DependsOn.size()}, Key(Key) {
    initTrailingObjects(DependsOn);
  }

  const StateKey &getKey() const { return Key; }

private:
  StateKey Key;
};

class EventHierarchy {
  llvm::BumpPtrAllocator Allocator;

public:
  const StateEvent &
  addStateEvent(StateKey Key, mlir::Operation *Location,
                std::initializer_list<ParentEventPointer> DependsOn) {
    void *Mem = Allocator.Allocate(StateEvent::sizeToAlloc(DependsOn.size()),
                                   alignof(StateEvent));
    return *new (Mem) StateEvent(Key, Location, DependsOn);
  }

  const StateEvent &addStateEvent(StateKey Key, mlir::Operation *Location) {
    return addStateEvent(Key, Location, {});
  }

  const DataFlowEvent &
  addDataFlowEvent(mlir::Operation *Location,
                   std::initializer_list<ParentEventPointer> DependsOn) {
    void *Mem = Allocator.Allocate(DataFlowEvent::sizeToAlloc(DependsOn.size()),
                                   alignof(DataFlowEvent));
    return *new (Mem) DataFlowEvent(Location, DependsOn);
  }

  const DataFlowEvent &addDataFlowEvent(mlir::Operation *Location) {
    return addDataFlowEvent(Location, {});
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
