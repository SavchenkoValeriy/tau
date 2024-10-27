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

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/JSON.h>

#include <memory>
#include <utility>

namespace mlir {
class Operation;
class Value;
} // end namespace mlir

namespace mlir::func {
class FuncOp;
} // namespace mlir::func

namespace tau::core {

class AnalysisTracer;

class Serializer {
public:
  template <class T> llvm::json::Value serialize(const T &) const;

  Serializer();
  ~Serializer();

  Serializer(const Serializer &) = delete;
  Serializer &operator=(const Serializer &) = delete;
  Serializer(Serializer &&) = delete;
  Serializer &operator=(Serializer &&) = delete;

private:
  friend AnalysisTracer;

  class Implementation;
  std::unique_ptr<Implementation> PImpl;
};

#ifdef ANALYSIS_TRACER_ENABLED
class AnalysisTracer {
public:
  AnalysisTracer(mlir::func::FuncOp &Function);
  ~AnalysisTracer();

  AnalysisTracer(const AnalysisTracer &) = delete;
  AnalysisTracer &operator=(const AnalysisTracer &) = delete;
  AnalysisTracer(AnalysisTracer &&) = delete;
  AnalysisTracer &operator=(AnalysisTracer &&) = delete;

  template <class CheckerState, class MemoryState>
  void recordBeforeState(mlir::Operation *Op, CheckerState &State,
                         MemoryState &Memory) {
    recordState("before", Op, State, Memory);
  }

  template <class CheckerState, class MemoryState>
  void recordAfterState(mlir::Operation *Op, CheckerState &State,
                        MemoryState &Memory) {
    recordState("after", Op, State, Memory);
  }

private:
  template <class CheckerState, class MemoryState>
  void recordState(llvm::StringRef Name, mlir::Operation *Op,
                   CheckerState &State, MemoryState &Memory) {
    if (!shouldTrace())
      return;

    llvm::json::Object Entry;
    Entry["operation"] = getSerializer().serialize(Op);
    Entry["state"] = getSerializer().serialize(State);
    Entry["memory"] = getSerializer().serialize(Memory);
    Entry["kind"] = Name;
    addEvent(std::move(Entry));
  }

  bool shouldTrace() const;
  void addEvent(llvm::json::Value Event);
  Serializer &getSerializer();

  class Implementation;
  std::unique_ptr<Implementation> PImpl;
};
#else
// Nop implementation if tracing is not enabled
class AnalysisTracer {
public:
  AnalysisTracer(mlir::func::FuncOp &Function){};

  template <class CheckerState, class MemoryState>
  void recordBeforeState(mlir::Operation *Op, CheckerState &State,
                         MemoryState &Memory) {}

  template <class CheckerState, class MemoryState>
  void recordAfterState(mlir::Operation *Op, CheckerState &State,
                        MemoryState &Memory) {}
};
#endif

template <>
llvm::json::Value Serializer::serialize(mlir::Operation *const &Op) const;
template <>
llvm::json::Value Serializer::serialize(const mlir::Value &Value) const;

} // namespace tau::core
