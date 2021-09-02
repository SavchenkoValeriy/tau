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

Type getPointee(Type Pointer) {
  if (auto AsPointer = Pointer.dyn_cast<AirPointerType>())
    return AsPointer.getElementType();
  return {};
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

//===----------------------------------------------------------------------===//
//                                    LoadOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &P, LoadOp &Op) {
  P << "air.load " << Op.from() << " : " << Op.from().getType();
}

static ParseResult parseLoad(OpAsmParser &Parser, OperationState &Result) {
  OpAsmParser::OperandType From;
  Type FromType;
  if (Parser.parseOperand(From) || Parser.parseColon() ||
      Parser.parseType(FromType) ||
      Parser.resolveOperand(From, FromType, Result.operands))
    return failure();

  if (Type ResultType = getPointee(FromType))
    return Parser.addTypeToList(ResultType, Result.types);

  return failure();
}

#define GET_OP_CLASSES
#include "tau/AIR/AirOps.cpp.inc"
