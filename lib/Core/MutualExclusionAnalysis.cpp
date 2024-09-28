#include "tau/Core/MutualExclusionAnalysis.h"
#include "tau/Core/TopoOrderEnumerator.h"

#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/Pass/AnalysisManager.h>

namespace llvm {
template <> struct GraphTraits<mlir::Block *> {
  using NodeRef = mlir::Block *;
  using ChildIteratorType = mlir::Block::succ_iterator;

  static NodeRef getEntryNode(mlir::Block *BB) { return BB; }
  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->succ_begin();
  }
  static inline ChildIteratorType child_end(NodeRef N) { return N->succ_end(); }
};
} // end namespace llvm

namespace tau::core {

MutualExclusionAnalysis::MutualExclusionAnalysis(mlir::Operation *Op,
                                                 mlir::AnalysisManager &AM)
    : TopoOrder(AM.getAnalysis<TopoOrderBlockEnumerator>()) {
  computeReachabilityMatrix(Op);
}

bool MutualExclusionAnalysis::areMutuallyExclusive(const mlir::Block *A,
                                                   const mlir::Block *B) const {
  const unsigned IdxA = TopoOrder.getTopoOrderIndex(A);
  const unsigned IdxB = TopoOrder.getTopoOrderIndex(B);
  return !ReachabilityMatrix[IdxA].test(IdxB) &&
         !ReachabilityMatrix[IdxB].test(IdxA);
}

bool MutualExclusionAnalysis::areMutuallyExclusive(mlir::Operation *A,
                                                   mlir::Operation *B) const {
  return areMutuallyExclusive(A->getBlock(), B->getBlock());
}

void MutualExclusionAnalysis::computeReachabilityMatrix(mlir::Operation *Op) {
  auto Function = mlir::cast<mlir::func::FuncOp>(Op);
  unsigned NumBlocks = Function.getBlocks().size();
  ReachabilityMatrix.resize(NumBlocks);

  // Initialize reachability matrix
  for (mlir::Block &Block : Function.getBlocks()) {
    const unsigned BlockIdx = TopoOrder.getTopoOrderIndex(&Block);
    ReachabilityMatrix[BlockIdx].resize(NumBlocks);
    // For the purposes of this analysis we consider that the block
    // is reachable from itself.
    ReachabilityMatrix[BlockIdx].set(BlockIdx);

    for (mlir::Block *Succ : Block.getSuccessors()) {
      const unsigned SuccIdx = TopoOrder.getTopoOrderIndex(Succ);
      ReachabilityMatrix[BlockIdx].set(SuccIdx);
    }
  }

  // Process SCCs and update reachability
  for (auto It = llvm::scc_begin(&Function.getBlocks().front()); !It.isAtEnd();
       ++It) {
    const auto &SCC = *It;
    if (SCC.empty())
      continue;

    // All the nodes within one SCC are guaranteed to have exactly the same
    // set of reachable nodes. For this reason, we choose one node out of the
    // whole SCC to be a representative.
    mlir::Block *RepresentativeBlock = SCC.front();
    const unsigned RepIdx = TopoOrder.getTopoOrderIndex(RepresentativeBlock);

    // First, let's merge all the nodes reachable from other nodes in the SCC,
    // so that we can do transitive closure over those.
    for (mlir::Block *OtherNodeInSCC : llvm::drop_begin(SCC)) {
      const unsigned BlockIdx = TopoOrder.getTopoOrderIndex(OtherNodeInSCC);
      ReachabilityMatrix[RepIdx] |= ReachabilityMatrix[BlockIdx];
    }

    // Calculate everything reachable from the representative node.
    for (unsigned SuccIndex : llvm::seq(0u, NumBlocks)) {
      if (ReachabilityMatrix[RepIdx].test(SuccIndex)) {
        ReachabilityMatrix[RepIdx] |= ReachabilityMatrix[SuccIndex];
      }
    }

    // And finally, set the same reachability set for the rest of the SCC nodes.
    for (mlir::Block *OtherNodeInSCC : llvm::drop_begin(SCC)) {
      const unsigned BlockIdx = TopoOrder.getTopoOrderIndex(OtherNodeInSCC);
      ReachabilityMatrix[BlockIdx] = ReachabilityMatrix[RepIdx];
    }
  }
}

} // namespace tau::core
