//===- Checker.h - Checker base classes -------------------------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
// This file defines the base classes and concepts for implementing static
// analysis checkers.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tau/AIR/AirAttrs.h"
#include "tau/Core/State.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <concepts>
#include <memory>

namespace tau::core {

class AbstractChecker {
public:
  /// Process the given operation.
  virtual void process(mlir::Operation *) = 0;

  /// Emit error on the given operation.
  virtual mlir::InFlightDiagnostic emitError(mlir::Operation *,
                                             air::StateID) const = 0;

  /// Emit issue note for the given operation.
  virtual void emitNote(mlir::InFlightDiagnostic &, mlir::Operation *,
                        air::StateID) const = 0;

  /// Returns the unique identifier that corresponds to this pass.
  mlir::TypeID getTypeID() const { return CheckerID; }

  /// Return the name of the checker
  virtual llvm::StringRef getName() const = 0;

  /// Return the command line argument/unique ID used when registering this
  /// checker.
  virtual llvm::StringRef getArgument() const = 0;

  /// Return the command line description used when registering this checker.
  virtual llvm::StringRef getDescription() const { return ""; }

  virtual ~AbstractChecker() = default;

protected:
  explicit AbstractChecker(mlir::TypeID CheckerID) : CheckerID(CheckerID) {}

private:
  mlir::TypeID CheckerID;
};

template <class CheckerT, ConcreteState State,
          StateMachine<State> CheckerStateMachine, class... Ops>
class CheckerBase : public AbstractChecker {
public:
  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const AbstractChecker *ToTest) {
    return ToTest->getTypeID() == mlir::TypeID::get<CheckerT>();
  }

  static_assert(State::getNumberOfErrorStates() != 0,
                "Checkers must have at least one error state");

  CheckerBase() : AbstractChecker(mlir::TypeID::get<CheckerT>()) {}
  CheckerBase(const CheckerBase &) = default;

  /// Mark the given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param What -- operand to change the state
  /// @param To -- target state to change the operand to
  ///
  /// Since this version doesn't specify which state the value should be in
  /// before, it will change any value to the @p To state only if it wasn't in
  /// any state.
  template <State To>
  void markChange(mlir::Operation *Where, mlir::Value What) const {
    static_assert(CheckerStateMachine.isInitial(To),
                  "This state is not mark as initial");
    unsigned WhatIdx = findOperand(Where, What);
    markChangeImpl(Where, WhatIdx, To.ID);
  }

  /// Mark the result of given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param To -- target state to change the operand to
  ///
  /// Since this version doesn't specify which state the value should be in
  /// before, it will change any value to the @p To state only if it wasn't in
  /// any state.
  template <State To> void markResultChange(mlir::Operation *Where) const {
    static_assert(CheckerStateMachine.isInitial(To),
                  "This state is not mark as initial");
    unsigned ResultIdx = getResultIdx(Where);
    markChangeImpl(Where, ResultIdx, To.ID);
  }

  /// Mark the given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param What -- operand to change the state
  /// @param From -- we only change the state of the value if it
  /// is in this state
  /// @param To -- target state to change the operand to
  template <State From, State To>
  void markChange(mlir::Operation *Where, mlir::Value What) const {
    static_assert(CheckerStateMachine.hasEdge(From, To),
                  "State machine doesn't have this transition");
    unsigned WhatIdx = findOperand(Where, What);
    markChangeImpl(Where, WhatIdx, From.ID, To.ID);
  }

  /// Mark the result of given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param From -- we only change the state of the value if it
  /// is in this state
  /// @param To -- target state to change the operand to
  template <State From, State To>
  void markResultChange(mlir::Operation *Where) const {
    static_assert(CheckerStateMachine.hasEdge(From, To),
                  "State machine doesn't have this transition");
    unsigned WhatIdx = getResultIdx(Where);
    markChangeImpl(Where, WhatIdx, From.ID, To.ID);
  }

  void process(mlir::Operation *) override;

  mlir::InFlightDiagnostic emitError(mlir::Operation *Op,
                                     air::StateID ID) const final {
    return getDerived().emitError(Op, ID);
  }

  void emitNote(mlir::InFlightDiagnostic &Diag, mlir::Operation *Op,
                air::StateID ID) const final {
    getDerived().emitNote(Diag, Op, ID);
  }

private:
  CheckerT &getDerived() { return *static_cast<CheckerT *>(this); }
  const CheckerT &getDerived() const {
    return *static_cast<const CheckerT *>(this);
  }

  using TypeSwitch = llvm::TypeSwitch<mlir::Operation *, void>;
  template <class OperationToProcess, class... RestOps>
  TypeSwitch &processAsDerived(TypeSwitch &Switch) {
    TypeSwitch &One = processOneAsDerived<OperationToProcess>(Switch);
    if constexpr (sizeof...(RestOps) == 0)
      return One;
    else
      return processAsDerived<RestOps...>(One);
  }

  template <class OperationToProcess>
  TypeSwitch &processOneAsDerived(TypeSwitch &Switch) {
    return Switch.Case<OperationToProcess>(
        [this](OperationToProcess Op) { getDerived().process(Op); });
  }

  template <class... Args>
  void markChangeImpl(mlir::Operation *Where, Args &&...Rest) const {
    mlir::MLIRContext *Context = Where->getContext();

    llvm::SmallVector<mlir::Attribute, 4> NewStateAttributes;
    if (auto StateArrayAttr =
            Where->getAttrOfType<mlir::ArrayAttr>(StateAttrID))
      NewStateAttributes.insert(NewStateAttributes.end(),
                                StateArrayAttr.begin(), StateArrayAttr.end());

    NewStateAttributes.push_back(air::StateChangeAttr::get(
        Context, getArgument(), std::forward<Args>(Rest)...));

    Where->setAttr(StateAttrID,
                   mlir::ArrayAttr::get(Context, NewStateAttributes));
  }

  static unsigned findOperand(mlir::Operation *Where, mlir::Value What) {
    auto AllOperands = Where->getOperands();
    auto It = llvm::find(AllOperands, What);
    if (It != AllOperands.end())
      return It - AllOperands.begin();

    // All operations in Air have exactly one result
    assert(What == Where->getOpResult(0));
    return getResultIdx(Where);
  }

  static unsigned getResultIdx(mlir::Operation *Of) {
    assert(Of->getNumResults() == 1 &&
           "The given operation should have a result to begin with");
    return Of->getNumOperands();
  }
};

template <class CheckerT, ConcreteState State,
          StateMachine<State> CheckerStateMachine, class... Ops>
void CheckerBase<CheckerT, State, CheckerStateMachine, Ops...>::process(
    mlir::Operation *BaseOp) {
  TypeSwitch Switch(BaseOp);
  if constexpr (sizeof...(Ops) > 0)
    processAsDerived<Ops...>(Switch);
}

namespace detail {
template <class T, class Op>
concept OpProcessor = requires(const T &Candidate, Op ToProcess) {
  // Check that the checker has a process method for the operation
  // it requested.
  Candidate.process(ToProcess);
};

// Check that the checker has process methods for all the operations
// it requested.
template <class T, class... Ops>
concept OpsProcessor = (... && OpProcessor<T, Ops>);

template <class T> struct CheckerVerifier {
  // If it's not derived from CheckerBase - it's not a checker.
  static constexpr bool DerivedFromCheckerBase = false;
};

template <class T, ConcreteState State, StateMachine<State> CheckerStateMachine,
          class... Ops>
struct CheckerVerifier<CheckerBase<T, State, CheckerStateMachine, Ops...>> {
  static constexpr bool DerivedFromCheckerBase = true;

  static constexpr auto StateMachineProps = CheckerStateMachine.getProperties();
  static constexpr bool AllStatesReachable =
      StateMachineProps.AllStatesReachable;
  static constexpr bool AllInitialStatesHavePathToError =
      StateMachineProps.AllInitialStatesHavePathToError;
  static constexpr bool HasStateLoops = StateMachineProps.HasLoops;

  // Check that the checker has emit methods in place.
  static constexpr bool HasEmitMethods =
      requires(const T &Candidate, State S, mlir::Operation *Op,
               mlir::InFlightDiagnostic &Diag) {
    { Candidate.emitError(Op, S) } -> std::same_as<mlir::InFlightDiagnostic>;
    { Candidate.emitNote(Diag, Op, S) };
  };

  // And all the process methods for operations it requested.
  static constexpr bool HasAllProcessMethods = OpsProcessor<T, Ops...>;
};

// We cannot write specialization for each checker, and specialization
// for CheckerBase will never be chosen in favor of a more generic template.
// Instead we use these two functions to choose exactly the types derived
// from CheckerBase.  For all other types, it will choose variadic function
// that has the lowest priority in terms of overloads.
CheckerVerifier<void> checkerDetector(...);

template <class T, ConcreteState State, StateMachine<State> SFA, class... Ops>
CheckerVerifier<CheckerBase<T, State, SFA, Ops...>>
checkerDetector(CheckerBase<T, State, SFA, Ops...> &&);
} // end namespace detail

template <class T,
          class Traits = decltype(detail::checkerDetector(std::declval<T>()))>
concept Checker = Traits::DerivedFromCheckerBase && Traits::HasEmitMethods &&
                  Traits::HasAllProcessMethods && Traits::AllStatesReachable &&
                  !Traits::HasStateLoops &&
                  Traits::AllInitialStatesHavePathToError;

} // end namespace tau::core
