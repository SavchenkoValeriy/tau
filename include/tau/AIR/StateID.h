//===- StateID.h - Raw state representation ---------------------*- C++ -*-===//
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

#include <llvm/ADT/Hashing.h>

namespace tau::air {
struct StateID {
  using Raw = unsigned;

  unsigned ID;

  static constexpr StateID fromRaw(Raw From) { return StateID{From}; }
  constexpr operator Raw() const { return ID; }

  static constexpr StateID getNonErrorState(unsigned Idx) {
    return fromRaw((Idx + 1) << 1);
  }

  static constexpr StateID getErrorState(unsigned Idx) {
    return fromRaw(((Idx + 1) << 1) | ErrorMask);
  }

  static constexpr bool isError(Raw ToTest) { return ToTest & ErrorMask; }
  constexpr bool isError() const { return isError(ID); }

private:
  static constexpr unsigned ErrorMask = 1;
};

constexpr inline bool operator==(const StateID &LHS, const StateID &RHS) {
  return StateID::Raw(LHS) == RHS;
}

constexpr inline bool operator!=(const StateID &LHS, const StateID &RHS) {
  return !(LHS == RHS);
}

} // end namespace tau::air

namespace llvm {
inline hash_code
hash_value(tau::air::StateID ID) { // NOLINT(readability-identifier-naming)
  return hash_value(tau::air::StateID::Raw(ID));
}
} // end namespace llvm
