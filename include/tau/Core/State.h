//===- State.h - Value state ------------------------------------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
// This file defines the State class template and related concepts, which are
// fundamental to representing and managing the checker-specific states
// encapsulating various astract properties.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tau/AIR/StateID.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <queue>
#include <utility>

namespace mlir {
class Attribute;
class Operation;
} // end namespace mlir

namespace tau {
namespace core {

constexpr llvm::StringLiteral StateAttrID = "state";

llvm::ArrayRef<mlir::Attribute> getStateAttributes(mlir::Operation *Op);

template <unsigned NumberOfNonErrorStates, unsigned NumberOfErrorStates>
class State {
public:
  /* implicit */ constexpr State(air::StateID From) : ID(From) {}
  explicit constexpr operator air::StateID() const { return ID; }

  /// Overall number of states
  static constexpr unsigned NumberOfStates =
      NumberOfNonErrorStates + NumberOfErrorStates;

  /// Get a non-error state by its index
  static constexpr State getNonErrorState(unsigned Idx) {
    assert(Idx < NumberOfNonErrorStates &&
           "We don't have that many non-error states");
    return air::StateID::getNonErrorState(Idx);
  }

  /// Get an error state by its index
  static constexpr State getErrorState(unsigned Idx) {
    assert(Idx < NumberOfErrorStates && "We don't have that many error states");
    return air::StateID::getErrorState(Idx);
  }

  /// Return a maximal possible state ID for this State.
  ///
  /// NOTE: this is an implementation detail and shouldn't be used
  ///       by checkers.
  static constexpr air::StateID::Raw getMaxStateID() {
    air::StateID::Raw MaxNonError = 0;
    if constexpr (NumberOfNonErrorStates != 0)
      MaxNonError = getNonErrorState(NumberOfNonErrorStates - 1).ID;
    const air::StateID::Raw MaxError =
        getErrorState(NumberOfErrorStates - 1).ID;
    return std::max(MaxNonError, MaxError);
  }

  static constexpr unsigned getNumberOfNonErrorStates() {
    return NumberOfNonErrorStates;
  }

  static constexpr unsigned getNumberOfErrorStates() {
    return NumberOfErrorStates;
  }

  /// Get an array of all possible states.
  static constexpr std::array<State, NumberOfStates> getAllStates() {
    // State doesn't have a default constructor, so all states
    // in the result array should be passed into the array initializer.
    return getAllStatesImpl(std::make_index_sequence<NumberOfNonErrorStates>(),
                            std::make_index_sequence<NumberOfErrorStates>());
  }

  /// Get an array of all possible non-error states.
  static constexpr std::array<State, NumberOfNonErrorStates>
  getAllNonErrorStates() {
    return getAllStatesWith<NumberOfNonErrorStates, getNonErrorState>();
  }

  /// Get an array of all possible non-error states.
  static constexpr std::array<State, NumberOfErrorStates> getAllErrorStates() {
    return getAllStatesWith<NumberOfErrorStates, getErrorState>();
  }

  /// Check if this state represents an error.
  constexpr bool isError() const { return ID.isError(); }

  air::StateID ID;

private:
  template <unsigned Size, auto getState>
  static constexpr std::array<State, Size> getAllStatesWith() {
    // State doesn't have a default cosntructor, so all states
    // in the result array should be passed into the array initializer.
    return getAllStatesWithImpl<Size, getState>(
        std::make_index_sequence<Size>());
  }

  template <unsigned Size, auto getState, std::size_t... Idx>
  static constexpr std::array<State, Size>
  getAllStatesWithImpl(std::index_sequence<Idx...>) {
    return {getState(Idx)...};
  }

  template <std::size_t... NonErrorIdx, std::size_t... ErrorIdx>
  static constexpr std::array<State, NumberOfStates>
  getAllStatesImpl(std::index_sequence<NonErrorIdx...>,
                   std::index_sequence<ErrorIdx...>) {
    const auto NonErrorStates = getAllNonErrorStates();
    const auto ErrorStates = getAllErrorStates();
    return {NonErrorStates[NonErrorIdx]..., ErrorStates[ErrorIdx]...};
  }

  static constexpr unsigned ErrorMask = 1;
};

/// A little helper concept to match concrete State instantiations.
template <class T>
concept ConcreteState = requires(T X) {
  { State(X) } -> std::same_as<T>;
};

/// A compile-time graph representing a state machine.
///
/// States and state transitions are at the core of the checker logic.
/// However, without proper checks, it is easy to make a mistake.
/// This class implements checks for some of the core requirements
/// to prevent logical errors.
template <ConcreteState State> class StateMachine {
public:
  using StateType = State;

  // This is definitely more than the number of states, but allows us
  // to use state IDs as indices.
  constexpr static auto Size = State::getMaxStateID() + 1;

  constexpr StateMachine() {
    Edges.fill(false);
    Initial.fill(false);
  }

  /// Add a transition between the given states.
  constexpr void addEdge(State From, State To) {
    assert(!From.isError() && "Error states should always be final");
    Edges[getEdgeIndex(From, To)] = true;
  }
  /// Mark the given state as an initial state of the automaton.
  ///
  /// NOTE: multiple states can be initial.
  constexpr void markInitial(State S) { Initial[S.ID] = true; }

  /// Check if the two given states have an edge between them.
  [[nodiscard]] constexpr bool hasEdge(State From, State To) const {
    return Edges[getEdgeIndex(From, To)];
  }

  /// Check if the given state is initial.
  [[nodiscard]] constexpr bool isInitial(State S) const {
    return Initial[S.ID];
  }

  /// Properties of the encoded state machine.
  struct Properties {
    /// True if all states of the automaton are reachable from initial states.
    bool AllStatesReachable = false;
    /// True if all initial states have a chain of transitions to some error.
    bool AllInitialStatesHavePathToError = true;
    /// True if there is a transition loop.
    bool HasLoops = false;
  };

  /// Return the properties of the encoded state machine.
  [[nodiscard]] consteval Properties getProperties() const {
    Properties Result;

    std::array<bool, Size> Visited;
    Visited.fill(false);

    const auto Traverse = [&](unsigned Index) {
      std::vector<unsigned> Stack;
      Stack.reserve(Size);

      std::array<bool, Size> OnStack, LocalVisited;
      OnStack.fill(false);
      LocalVisited.fill(false);

      const auto Push = [&Stack, &OnStack](unsigned Index) {
        Stack.push_back(Index);
        OnStack[Index] = true;
      };

      Push(Index);

      bool ReachesError = false;

      while (!Stack.empty()) {
        const unsigned CurrentIndex = Stack.back();

        // We keep a separate set of visited nodes for each traversal because we
        // want to also check if each initial state has a path to an error
        // state. This means that we can't interrupt visitation of a child state
        // just because we reached it from another initial state.
        LocalVisited[CurrentIndex] = true;

        bool HasAddedAnyChildren = false;
        // ReachesError is true when ANY of the visited states is an error
        // state.
        ReachesError |= air::StateID::isError(CurrentIndex);

        for (unsigned ChildIndex = 1; ChildIndex < Size; ++ChildIndex) {
          if (!Edges[CurrentIndex * Size + ChildIndex])
            continue;

          // If it's on the stack, this node has been visited as its child or
          // a transitive child meaning that this is a loop.
          if (OnStack[ChildIndex])
            Result.HasLoops = true;
          // Otherwise, we can order it for visiting if this traversal hasn't
          // done it already.
          else if (!LocalVisited[ChildIndex]) {
            HasAddedAnyChildren = true;
            Push(ChildIndex);
          }
        }

        // If we didn't add any nodes onto the stack, this node is at the top
        // of the stack and should be removed.
        if (!HasAddedAnyChildren) {
          Stack.pop_back();
          Visited[CurrentIndex] = true;
          OnStack[CurrentIndex] = false;
        }
      }

      // ALL initial states should have a path to error.
      Result.AllInitialStatesHavePathToError &= ReachesError;
    };

    for (unsigned I = 0; I != Size; ++I)
      if (Initial[I])
        Traverse(I);

    const auto AllStates = State::getAllStates();
    // Let's check that all states are reachable from initial states.
    Result.AllStatesReachable =
        std::all_of(AllStates.begin(), AllStates.end(),
                    [&Visited](State S) { return Visited[S.ID]; });

    return Result;
  }

  static constexpr unsigned getEdgeIndex(State From, State To) {
    const air::StateID::Raw FromIndex = From.ID, ToIndex = To.ID;
    return FromIndex * Size + ToIndex;
  }

  std::array<bool, Size * Size> Edges;
  std::array<bool, Size> Initial;
};

template <unsigned X, unsigned Y>
inline bool operator==(const State<X, Y> &LHS, const State<X, Y> &RHS) {
  return LHS.ID == RHS.ID;
}

template <unsigned X, unsigned Y>
inline bool operator!=(const State<X, Y> &LHS, const State<X, Y> &RHS) {
  return !(LHS == RHS);
}

} // end namespace core
} // end namespace tau
