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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/PointerIntPair.h>
#include <llvm/ADT/PointerUnion.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/TrailingObjects.h>

#include <initializer_list>
#include <memory>

namespace mlir {
class Operation;
} // end namespace mlir

namespace tau::core {

class DataFlowEvent;
class StateEvent;
class EventHierarchy;
class TopoOrderBlockEnumerator;

using AbstractEvent =
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
                  public llvm::TrailingObjects<Derived, AbstractEvent> {
  using TrailingObjects = llvm::TrailingObjects<Derived, AbstractEvent>;

  friend TrailingObjects;
  friend EventHierarchy;

  static size_t sizeToAlloc(size_t NumberOfParents) {
    return Derived::template totalSizeToAlloc<AbstractEvent>(NumberOfParents);
  }

protected:
  using CommonBase::CommonBase;

  size_t numTrailingObjects(
      TrailingObjects::template OverloadToken<AbstractEvent>) const {
    return getNumberOfParents();
  }

  Derived *getAsDerived() { return static_cast<Derived *>(this); }
  const Derived *getAsDerived() const {
    return static_cast<const Derived *>(this);
  }
  void initTrailingObjects(std::initializer_list<AbstractEvent> Pointers) {
    std::uninitialized_copy(
        Pointers.begin(), Pointers.end(),
        getAsDerived()->template getTrailingObjects<AbstractEvent>());
  }

public:
  /// @returns An array of parent event pointers.
  llvm::ArrayRef<const AbstractEvent> getParents() const {
    return {getAsDerived()->template getTrailingObjects<AbstractEvent>(),
            getNumberOfParents()};
  }
};

} // end namespace detail

/// @brief Represents a data flow event in the analysis.
class DataFlowEvent final : public detail::BaseEvent<DataFlowEvent> {
public:
  /// @brief Constructs a DataFlowEvent.
  /// @param Location The MLIR operation associated with this event.
  /// @param DependsOn The parent events this event depends on.
  DataFlowEvent(mlir::Operation *Location,
                std::initializer_list<AbstractEvent> DependsOn)
      : detail::BaseEvent<DataFlowEvent>{Location, DependsOn.size()} {
    initTrailingObjects(DependsOn);
  }
};

/// @brief Represents a key for identifying a state event.
struct StateKey {
  llvm::StringRef CheckerID;
  air::StateID State;

  bool operator==(const StateKey &Other) const {
    return CheckerID == Other.CheckerID && State == Other.State;
  }
};

/// @brief Represents a state event in the analysis.
class StateEvent final : public detail::BaseEvent<StateEvent> {
public:
  /// @brief Constructs a StateEvent.
  /// @param Key The key identifying this state event.
  /// @param Location The MLIR operation associated with this event.
  /// @param DependsOn The parent events this event depends on.
  StateEvent(StateKey Key, mlir::Operation *Location,
             std::initializer_list<AbstractEvent> DependsOn)
      : detail::BaseEvent<StateEvent>{Location, DependsOn.size()}, Key(Key) {
    initTrailingObjects(DependsOn);
  }

  /// @returns The key identifying this state event.
  const StateKey &getKey() const { return Key; }

private:
  StateKey Key;
};

/// @brief Manages the hierarchy of events in the analysis.
class EventHierarchy {
  llvm::BumpPtrAllocator Allocator;

public:
  /// @brief Adds a new StateEvent to the hierarchy.
  /// @returns A reference to the newly created StateEvent.
  const StateEvent &
  addStateEvent(StateKey Key, mlir::Operation *Location,
                std::initializer_list<AbstractEvent> DependsOn) {
    void *Mem = Allocator.Allocate(StateEvent::sizeToAlloc(DependsOn.size()),
                                   alignof(StateEvent));
    return *new (Mem) StateEvent(Key, Location, DependsOn);
  }

  /// @brief Adds a new StateEvent with no parents to the hierarchy.
  /// @returns A reference to the newly created StateEvent.
  const StateEvent &addStateEvent(StateKey Key, mlir::Operation *Location) {
    return addStateEvent(Key, Location, {});
  }

  /// @brief Adds a new DataFlowEvent to the hierarchy.
  /// @returns A reference to the newly created DataFlowEvent.
  const DataFlowEvent &
  addDataFlowEvent(mlir::Operation *Location,
                   std::initializer_list<AbstractEvent> DependsOn) {
    void *Mem = Allocator.Allocate(DataFlowEvent::sizeToAlloc(DependsOn.size()),
                                   alignof(DataFlowEvent));
    return *new (Mem) DataFlowEvent(Location, DependsOn);
  }

  /// @brief Adds a new DataFlowEvent with no parents to the hierarchy.
  /// @returns A reference to the newly created DataFlowEvent.
  const DataFlowEvent &addDataFlowEvent(mlir::Operation *Location) {
    return addDataFlowEvent(Location, {});
  }

  /// @brief Retrieves and sorts all parent events for a given event.
  /// @param Event The event to start from.
  /// @param Enumerator The topological order enumerator for basic blocks.
  /// @returns A vector of sorted events.
  llvm::SmallVector<AbstractEvent, 20>
  linearizeChainOfEvents(const AbstractEvent &Event,
                         const TopoOrderBlockEnumerator &Enumerator) const;
};

} // end namespace tau::core

namespace std {
template <> struct hash<tau::core::StateKey> {
  size_t operator()(const tau::core::StateKey &Key) const {
    return llvm::hash_combine(llvm::hash_value(Key.State), Key.CheckerID);
  }
};

} // end namespace std
