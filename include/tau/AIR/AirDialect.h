//===- AirDialect.h - AIR dialect -------------------------------*- C++ -*-===//
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

#include <mlir/IR/Dialect.h>

#include "tau/AIR/AirOpsDialect.h.inc"

namespace tau::air {
inline bool isCompatibleType(mlir::Type) {
  // For now, we consifer every type to be compatible with Air
  return true;
}
} // end namespace tau::air
