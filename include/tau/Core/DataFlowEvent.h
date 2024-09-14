//===- DataFlowEvent.h - Forest of data-flow events -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DataFlowEventForest class, which is responsible for
// managing memory and relating data-flow events.
//
// The DataFlowEventForest maintains a forest of events, where each tree
// in the forest represents a sequence of assignments. This structure
// allows the analysis to track the history of data-flow transitions and their
// relationships, which is crucial for explaining why we believe one
// value is equal to some other value.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tau/Core/EventForest.h"

namespace mlir {
class Operation;
} // end namespace mlir

namespace tau::core {

struct DataFlowEvent {
  mlir::Operation *Location = nullptr;
  const DataFlowEvent *Parent = nullptr;
};

class DataFlowEventForest : public EventForest<DataFlowEvent> {};

} // end namespace tau::core
