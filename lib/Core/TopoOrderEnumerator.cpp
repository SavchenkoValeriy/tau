#include "tau/Core/TopoOrderEnumerator.h"

#include <llvm/ADT/SCCIterator.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/RegionGraphTraits.h>

using namespace llvm;
using namespace mlir;

tau::core::TopoOrderBlockEnumerator::TopoOrderBlockEnumerator(Operation *Op) {
  assert(isa<FuncOp>(Op) &&
         "Topological order enumeration is only available for functions");

  FuncOp Function = cast<FuncOp>(Op);
  unsigned NextIndex = 0;

  for (auto SCCIt = scc_begin(&Function.getBody()); !SCCIt.isAtEnd(); ++SCCIt)
    for (const Block *BB : *SCCIt)
      Indices[BB] = NextIndex++;
}
