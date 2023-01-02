#include "tau/Core/PostOrderEnumerator.h"

#include <llvm/ADT/PostOrderIterator.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/RegionGraphTraits.h>

using namespace llvm;
using namespace mlir;
using namespace mlir::func;

tau::core::PostOrderBlockEnumerator::PostOrderBlockEnumerator(Operation *Op) {
  assert(isa<FuncOp>(Op) &&
         "Post-order enumeration is only available for functions");

  FuncOp Function = cast<FuncOp>(Op);
  unsigned NextIndex = 0;
  for (const Block *BB : post_order(&Function.getBody())) {
    Indices[BB] = NextIndex++;
  }
}
