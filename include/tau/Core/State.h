//===- State.h - Value state ------------------------------------*- C++ -*-===//
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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

#include <cassert>

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
  explicit operator air::StateID() const { return ID; }

  static constexpr State getNonErrorState(unsigned Idx) {
    assert(Idx < NumberOfNonErrorStates &&
           "We don't have that many non-error states");
    return air::StateID::getNonErrorState(Idx);
  }

  static constexpr State getErrorState(unsigned Idx) {
    assert(Idx < NumberOfErrorStates &&
           "We don't have that many non-error states");
    return air::StateID::getErrorState(Idx);
  }

  bool isError() const { return ID.isError(); }

  air::StateID ID;

private:
  static constexpr unsigned ErrorMask = 1;
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
