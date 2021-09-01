#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirTypes.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LogicalResult.h>

namespace {

using namespace mlir;
using namespace tau::air;

LogicalResult verify(StoreOp &Store) {
  Type WhatType = Store.what().getType();
  AirPointerType WhereType = Store.where().getType().cast<AirPointerType>();

  if (WhereType.getElementType() == WhatType)
    return success();

  return Store.emitError() << "type of stored value (" << WhatType
                           << ") doesn't match the pointer type (" << WhereType
                           << ")";
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//                                  ConstantOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &P, ConstantOp &Op) {
  P << "air.constant " << Op.getValue();
}

static ParseResult parseConstantOp(OpAsmParser &Parser,
                                   OperationState &Result) {
  Attribute ValueAttr;
  if (Parser.parseAttribute(ValueAttr, "value", Result.attributes))
    return failure();

  Type ConstType = ValueAttr.getType();

  return Parser.addTypeToList(ConstType, Result.types);
}

void ConstantIntOp::build(OpBuilder &Builder, OperationState &Result,
                          llvm::APInt Value, IntegerType Type) {
  ConstantOp::build(Builder, Result, Type, Builder.getIntegerAttr(Type, Value));
}

bool ConstantIntOp::classof(Operation *Op) {
  mlir::Type T = Op->getResult(0).getType();
  return ConstantOp::classof(Op) && T.isSignedInteger() ||
         T.isUnsignedInteger();
}

#define GET_OP_CLASSES
#include "tau/AIR/AirOps.cpp.inc"
