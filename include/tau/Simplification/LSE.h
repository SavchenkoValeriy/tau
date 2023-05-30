//===- LSE.h - Load-store elimination ---------------------------*- C++ -*-===//
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

#include <memory>

namespace mlir {
class Pass;
} // end namespace mlir

namespace tau::simple {
std::unique_ptr<mlir::Pass> createLSEPass();
} // end namespace tau::simple
