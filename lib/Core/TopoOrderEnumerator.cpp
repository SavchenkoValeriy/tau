#include "tau/Core/TopoOrderEnumerator.h"

#include <llvm/ADT/SCCIterator.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/RegionGraphTraits.h>

#include <stack>

using namespace llvm;
using namespace mlir;
using namespace mlir::func;

tau::core::TopoOrderBlockEnumerator::TopoOrderBlockEnumerator(Operation *Op) {
  assert(isa<FuncOp>(Op) &&
         "Topological order enumeration is only available for functions");

  // Data structures required for enumeration
  llvm::DenseSet<Block *> Visited;
  std::stack<Block *> Stack;
  unsigned NextIndex = 0;

  FuncOp Function = cast<FuncOp>(Op);
  if (Function.getBlocks().empty())
    return;

  Block *Entry = &Function.getBlocks().front();
  Stack.push(Entry);

  while (!Stack.empty()) {
    Block *Current = Stack.top();

    bool AddedAnyChildren = false;
    for (Block *Succ : Current->getSuccessors())
      // Check if we ever pushed this block to stack.
      // It is either a forward edge to the block that we
      // got to using some other path, or a back edge, which
      // we ignore for our purposes.
      if (auto [It, IsNew] = Visited.insert(Succ); IsNew) {
        AddedAnyChildren = true;
        Stack.push(Succ);
      }

    if (!AddedAnyChildren) {
      Stack.pop();
      Indices[Current] = NextIndex++;
    }
  }

  // Now let's handle unreachable blocks. We'll give them the lowest
  // topological indices and shift all other indices by the number
  // of such blocks.
  const unsigned NumberOfUnreachable =
      // 1 is for the Entry block that nevere gets added to Visited
      Function.getBlocks().size() - Visited.size() - 1;
  if (NumberOfUnreachable != 0) {
    for (auto &[BB, Index] : Indices)
      Index += NumberOfUnreachable;

    unsigned NextIndex = 0;

    for (Block &BB : Function.getBlocks()) {
      if (!Visited.contains(&BB)) {
        Indices[&BB] = NextIndex++;
      }
    }
  }
}
