//===- Analysis.h - Analysis umbrella header --------------------*- C++ -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the main Tau analysis. All of the issue detection logic
//  is encapsulated and performed by this one pass.
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
