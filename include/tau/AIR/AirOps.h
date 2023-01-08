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

#include "tau/AIR/AirTypes.h"

#include <llvm/ADT/APSInt.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_OP_CLASSES
#include "tau/AIR/AirOps.h.inc"

namespace tau::air {

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of IntegerType.
///
///   %1 = "air.constant"(){value: 42} : i32
///
class ConstantIntOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;
  /// Build a constant int op producing an integer with the specified type.
  static void build(mlir::OpBuilder &Builder, mlir::OperationState &Result,
                    llvm::APInt Value, mlir::IntegerType Type);

  /// Build a constant int op producing an integer with the specified type.
  static void build(mlir::OpBuilder &Builder, mlir::OperationState &Result,
                    int64_t Value, mlir::IntegerType Type);

  llvm::APSInt getValue() {
    return (*this)->getAttrOfType<mlir::IntegerAttr>("value").getAPSInt();
  }

  static bool classof(mlir::Operation *Op);
};

/// This is a refinement of the "constant" op for the case where it is
/// returning a float value of FloatType.
///
///   %1 = "air.constant"(){value: 42.0} : bf16
///
class ConstantFloatOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  /// Builds a constant float op producing a float of the specified type.
  static void build(mlir::OpBuilder &Builder, mlir::OperationState &Result,
                    llvm::APFloat Value, mlir::FloatType Type);

  /// Builds a constant float op producing a float of the specified type.
  static void build(mlir::OpBuilder &Builder, mlir::OperationState &Result,
                    double Value, mlir::FloatType Type);

  llvm::APFloat getValue() {
    return (*this)->getAttrOfType<mlir::FloatAttr>("value").getValue();
  }

  static bool classof(mlir::Operation *Op);
};

} // end namespace tau::air
