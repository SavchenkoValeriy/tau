//===- AddressTakenAnalysis.h - Address-taken analysis ----------*- C++ -*-===//
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

#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/Value.h>

namespace mlir {

class Operation;

} // end namespace mlir

namespace tau::core {

class AddressTakenAnalysis {
public:
  AddressTakenAnalysis(mlir::Operation *Function);

  [[nodiscard]] bool hasAddressBeenTaken(mlir::Value) const;
  [[nodiscard]] const llvm::DenseSet<mlir::Value> &
  getAddressTakenValues() const {
    return AddressTakenValues;
  }

private:
  llvm::DenseSet<mlir::Value> AddressTakenValues;
};

} // end namespace tau::core
