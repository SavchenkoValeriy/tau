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

#include <cassert>

namespace tau {
namespace core {

template <unsigned NumberOfNonErrorStates, unsigned NumberOfErrorStates>
class State {
public:
  using ID = unsigned;

  /* implicit */ constexpr State(ID From) : StateID(From) {}
  operator ID() { return StateID; }

  static constexpr State getNonErrorState(unsigned Idx) {
    assert(Idx < NumberOfNonErrorStates &&
           "We don't have that many non-error states");
    return (Idx + 1) << 1;
  }

  static constexpr State getErrorState(unsigned Idx) {
    assert(Idx < NumberOfErrorStates &&
           "We don't have that many non-error states");
    return ((Idx + 1) << 1) | ErrorMask;
  }

  static constexpr bool isError(ID ToTest) { return ToTest & ErrorMask; }
  bool isError() const { return isError(StateID); }

private:
  ID StateID;
  static constexpr unsigned ErrorMask = 1;
};

} // end namespace core
} // end namespace tau
