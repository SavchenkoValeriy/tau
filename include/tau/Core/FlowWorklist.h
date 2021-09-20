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

#include "tau/Core/PostOrderEnumerator.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/PriorityQueue.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>

namespace tau::core {

namespace detail {

template <class Derived, class Comparator> class WorklistBase {
protected:
  PostOrderBlockEnumerator &Enumeration;

private:
  llvm::BitVector EnqueuedBlocks;
  using QueueImpl = llvm::SmallVector<const mlir::Block *, 20>;
  llvm::PriorityQueue<const mlir::Block *, QueueImpl, Comparator> WorkList;

  Derived &getDerived() { return *static_cast<Derived *>(this); }

public:
  WorklistBase(mlir::FuncOp Function, mlir::AnalysisManager &AM)
      : Enumeration(AM.getAnalysis<PostOrderBlockEnumerator>()),
        EnqueuedBlocks(Function.getBlocks().size()),
        WorkList(this->getDerived().getComparator()) {}

  void enqueue(const mlir::Block *BB) {
    unsigned BBIndex = Enumeration.getPostOrderIndex(BB);

    if (!EnqueuedBlocks[BBIndex]) {
      EnqueuedBlocks[BBIndex] = true;
      WorkList.push(BB);
    }
  }

  const mlir::Block *dequeue() {
    if (WorkList.empty())
      return nullptr;

    const mlir::Block *BB = WorkList.top();
    WorkList.pop();
    EnqueuedBlocks[Enumeration.getPostOrderIndex(BB)] = false;
    return BB;
  }
};

struct ForwardTraversalComparator {
  PostOrderBlockEnumerator &Enumerator;
  bool operator()(const mlir::Block *LHS, const mlir::Block *RHS) const {
    return Enumerator.getPostOrderIndex(LHS) >
           Enumerator.getPostOrderIndex(RHS);
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
};

} // end namespace tau::core
