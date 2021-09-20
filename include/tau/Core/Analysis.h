//===- Analysis.h - Analysis umbrella header --------------------*- C++ -*-===//
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

namespace tau {
namespace core {
std::unique_ptr<mlir::Pass> createMainAnalysis();
} // end namespace core
} // end namespace tau
