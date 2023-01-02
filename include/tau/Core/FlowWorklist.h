//===- FlowWorklist.h - Worklist for dataflow algorithms --------*- C++ -*-===//
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

#include "tau/Core/TopoOrderEnumerator.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/PriorityQueue.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>

namespace tau::core {

namespace detail {

template <class Derived, class Comparator> class WorklistBase {
protected:
  TopoOrderBlockEnumerator &Enumeration;

private:
  llvm::BitVector EnqueuedBlocks;
  using QueueImpl = llvm::SmallVector<mlir::Block *, 20>;
  llvm::PriorityQueue<mlir::Block *, QueueImpl, Comparator> WorkList;

  Derived &getDerived() { return *static_cast<Derived *>(this); }

public:
  WorklistBase(mlir::Operation *Function, mlir::AnalysisManager &AM)
      : Enumeration(AM.getAnalysis<TopoOrderBlockEnumerator>()),
        EnqueuedBlocks(
            llvm::cast<mlir::func::FuncOp>(Function).getBlocks().size()),
        WorkList(this->getDerived().getComparator()) {}

  void enqueue(mlir::Block *BB) {
    unsigned BBIndex = Enumeration.getTopoOrderIndex(BB);

    if (!EnqueuedBlocks[BBIndex]) {
      EnqueuedBlocks[BBIndex] = true;
      WorkList.push(BB);
    }
  }

  mlir::Block *dequeue() {
    if (WorkList.empty())
      return nullptr;

    mlir::Block *BB = WorkList.top();
    WorkList.pop();
    EnqueuedBlocks[Enumeration.getTopoOrderIndex(BB)] = false;
    return BB;
  }
};

struct ForwardTraversalComparator {
  TopoOrderBlockEnumerator &Enumerator;
  bool operator()(const mlir::Block *LHS, const mlir::Block *RHS) const {
    return Enumerator.getTopoOrderIndex(LHS) <
           Enumerator.getTopoOrderIndex(RHS);
  }
};

struct BackwardTraversalComparator {
  ForwardTraversalComparator ReversedComparator;
  bool operator()(const mlir::Block *LHS, const mlir::Block *RHS) const {
    return ReversedComparator(RHS, LHS);
  }
};

} // end namespace detail

class ForwardWorklist
    : public detail::WorklistBase<ForwardWorklist,
                                  detail::ForwardTraversalComparator> {
public:
  using Base =
      detail::WorklistBase<ForwardWorklist, detail::ForwardTraversalComparator>;
  using Base::WorklistBase;

  detail::ForwardTraversalComparator getComparator() const {
    return {Enumeration};
  }

  void enqueueSuccessors(mlir::Block &BB) {
    for (mlir::Block *Succ : BB.getSuccessors())
      enqueue(Succ);
  }
};

class BackwardWorklist
    : public detail::WorklistBase<BackwardWorklist,
                                  detail::BackwardTraversalComparator> {
public:
  using Base = detail::WorklistBase<BackwardWorklist,
                                    detail::BackwardTraversalComparator>;
  using Base::WorklistBase;

  detail::BackwardTraversalComparator getComparator() const {
    return {Enumeration};
  }

  void enqueuePredecessors(mlir::Block &BB) {
    for (mlir::Block *Pred : BB.getPredecessors())
      enqueue(Pred);
  }
};

} // end namespace tau::core
