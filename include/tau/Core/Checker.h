//===- Checker.h - Checker base classes -------------------------*- C++ -*-===//
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

#include <memory>

namespace tau::core {

class Checker {
public:
  /// Process the given operation.
  virtual void process(mlir::Operation *) = 0;

  /// Emit error on the given operation.
  virtual mlir::InFlightDiagnostic emitError(mlir::Operation *,
                                             air::StateID) = 0;

  /// Emit issue note for the given operation.
  virtual void emitNote(mlir::InFlightDiagnostic &, mlir::Operation *,
                        air::StateID) = 0;

  /// Returns the unique identifier that corresponds to this pass.
  mlir::TypeID getTypeID() const { return CheckerID; }

  /// Return the name of the checker
  virtual llvm::StringRef getName() const = 0;

  /// Return the command line argument/unique ID used when registering this
  /// checker.
  virtual llvm::StringRef getArgument() const = 0;

  /// Return the command line description used when registering this checker.
  virtual llvm::StringRef getDescription() const { return ""; }

  virtual ~Checker() = default;

protected:
  explicit Checker(mlir::TypeID CheckerID) : CheckerID(CheckerID) {}

private:
  mlir::TypeID CheckerID;
};

template <class CheckerT, class State, class... Ops>
class CheckerWrapper : public Checker {
public:
  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const Checker *ToTest) {
    return ToTest->getTypeID() == mlir::TypeID::get<CheckerT>();
  }

  CheckerWrapper() : Checker(mlir::TypeID::get<CheckerT>()) {}
  CheckerWrapper(const CheckerWrapper &) = default;

  /// Mark the given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param What -- operand to change the state
  /// @param To -- target state to change the operand to
  ///
  /// Since this version doesn't specify which state the value should be in
  /// before, it will change any value to the @p To state only if it wasn't in
  /// any state.
  void markChange(mlir::Operation *Where, mlir::Value What, State To) const {
    unsigned WhatIdx = findOperand(Where, What);
    markChangeImpl(Where, WhatIdx, To);
  }

  /// Mark the result of given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param To -- target state to change the operand to
  ///
  /// Since this version doesn't specify which state the value should be in
  /// before, it will change any value to the @p To state only if it wasn't in
  /// any state.
  void markResultChange(mlir::Operation *Where, State To) const {
    unsigned ResultIdx = getResultIdx(Where);
    markChangeImpl(Where, ResultIdx, To);
  }

  /// Mark the given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param What -- operand to change the state
  /// @param From -- we only change the state of the value if it
  /// is in this state
  /// @param To -- target state to change the operand to
  void markChange(mlir::Operation *Where, mlir::Value What, State From,
                  State To) const {
    unsigned WhatIdx = findOperand(Where, What);
    markChangeImpl(Where, WhatIdx, From, To);
  }

  /// Mark the result of given operation as changing the state.
  ///
  /// @param Where -- operation to be marked
  /// @param From -- we only change the state of the value if it
  /// is in this state
  /// @param To -- target state to change the operand to
  void markResultChange(mlir::Operation *Where, State From, State To) const {
    unsigned WhatIdx = getResultIdx(Where);
    markChangeImpl(Where, WhatIdx, From, To);
  }

  void process(mlir::Operation *) override;

  mlir::InFlightDiagnostic emitError(mlir::Operation *Op,
                                     air::StateID ID) final {
    return getDerived().emitError(Op, ID);
  }

  void emitNote(mlir::InFlightDiagnostic &Diag, mlir::Operation *Op,
                air::StateID ID) final {
    getDerived().emitNote(Diag, Op, ID);
  }

private:
  CheckerT &getDerived() { return *static_cast<CheckerT *>(this); }

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

template <class CheckerT, class State, class... Ops>
void CheckerWrapper<CheckerT, State, Ops...>::process(mlir::Operation *BaseOp) {
  TypeSwitch Switch(BaseOp);
  processAsDerived<Ops...>(Switch);
}

} // end namespace tau::core
