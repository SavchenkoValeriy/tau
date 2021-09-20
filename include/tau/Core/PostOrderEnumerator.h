//===- PostOrderEnumerator.h - Enumerator for basic blocks ------*- C++ -*-===//
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

#include <llvm/ADT/DenseMap.h>

namespace mlir {

class Block;
class Operation;

} // end namespace mlir

namespace tau::core {

class PostOrderBlockEnumerator {
public:
  PostOrderBlockEnumerator(mlir::Operation *Function);

  unsigned getPostOrderIndex(const mlir::Block *BB) {
    assert(Indices.count(BB) == 1 &&
           "All reachable blocks should be enumerated");
    return Indices[BB];
  }

private:
  llvm::DenseMap<const mlir::Block *, unsigned> Indices;
};

} // end namespace tau::core
