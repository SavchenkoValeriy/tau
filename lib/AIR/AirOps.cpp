#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirTypes.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LogicalResult.h>

namespace {

using namespace mlir;
using namespace tau::air;

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

//===----------------------------------------------------------------------===//
//                                   StoreOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(StoreOp &Store) {
  Type WhatType = Store.what().getType();
  AirPointerType WhereType = Store.where().getType().cast<AirPointerType>();

  if (WhereType.getElementType() == WhatType)
    return success();

  return Store.emitError() << "type of stored value (" << WhatType
                           << ") doesn't match the pointer type (" << WhereType
                           << ")";
}

static void print(OpAsmPrinter &P, StoreOp &Op) {
  P << "air.store " << Op.what() << " -> " << Op.where() << " : "
    << Op.where().getType();
}

static ParseResult parseStore(OpAsmParser &Parser, OperationState &Result) {
  OpAsmParser::OperandType What, Where;
  Type WhereType;
  if (Parser.parseOperand(What) || Parser.parseArrow() ||
      Parser.parseOperand(Where) || Parser.parseColon())
    return failure();

  Type WhatType = getPointee(WhereType);
  if (!WhatType || Parser.resolveOperand(What, WhatType, Result.operands) ||
      Parser.resolveOperand(Where, WhereType, Result.operands))
    return failure();

  return success();
}

#define GET_OP_CLASSES
#include "tau/AIR/AirOps.cpp.inc"
