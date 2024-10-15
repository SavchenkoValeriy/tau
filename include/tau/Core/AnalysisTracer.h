//===- AnalysisTracer.h - Trace data-flow analysis --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
///  This file defines the tracer component that tracks intermediate steps of
///  of the analysis and serializes them into JSON for further visualization.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace mlir::func {
class FuncOp;
} // namespace mlir::func

namespace tau::core {

class AnalysisTracer {
public:
  AnalysisTracer(mlir::func::FuncOp &Function);
  ~AnalysisTracer();

  AnalysisTracer(const AnalysisTracer &) = delete;
  AnalysisTracer &operator=(const AnalysisTracer &) = delete;
  AnalysisTracer(AnalysisTracer &&) = delete;
  AnalysisTracer &operator=(AnalysisTracer &&) = delete;

private:
  class Implementation;
  std::unique_ptr<Implementation> PImpl;
};

} // namespace tau::core
