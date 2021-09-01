//===- AirOps.h - AIR operations --------------------------------*- C++ -*-===//
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

#include <llvm/ADT/APSInt.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_OP_CLASSES
#include "tau/AIR/AirOps.h.inc"

namespace tau::air {

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of IntegerType.
///
///   %1 = "std.constant"(){value: 42} : i32
///
class ConstantIntOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;
  /// Build a constant int op producing an integer with the specified type.
  static void build(mlir::OpBuilder &Builder, mlir::OperationState &Result,
                    llvm::APInt Value, mlir::IntegerType Type);

  llvm::APSInt getValue() {
    return (*this)->getAttrOfType<mlir::IntegerAttr>("value").getAPSInt();
  }

  static bool classof(mlir::Operation *Op);
};

} // end namespace tau::air
