#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirTypes.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LogicalResult.h>

namespace {

using namespace mlir;
using namespace tau::air;

Type getPointee(Type Pointer) {
  if (auto AsPointer = Pointer.dyn_cast<PointerType>())
    return AsPointer.getElementType();
  return {};
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//                                  ConstantOp
//===----------------------------------------------------------------------===//

void ConstantIntOp::build(OpBuilder &Builder, OperationState &Result,
                          llvm::APInt Value, IntegerType Type) {
  ConstantOp::build(Builder, Result, Type, Builder.getIntegerAttr(Type, Value));
}

void ConstantIntOp::build(OpBuilder &Builder, OperationState &Result,
                          int64_t Value, IntegerType Type) {
  ConstantOp::build(Builder, Result, Type, Builder.getIntegerAttr(Type, Value));
}

bool ConstantIntOp::classof(Operation *Op) {
  if (!ConstantOp::classof(Op))
    return false;

  mlir::Type T = Op->getResult(0).getType();
  return T.isSignedInteger() || T.isUnsignedInteger();
}

void ConstantFloatOp::build(OpBuilder &Builder, OperationState &Result,
                            llvm::APFloat Value, FloatType Type) {
  ConstantOp::build(Builder, Result, Type, Builder.getFloatAttr(Type, Value));
}

void ConstantFloatOp::build(OpBuilder &Builder, OperationState &Result,
                            double Value, FloatType Type) {
  ConstantOp::build(Builder, Result, Type, Builder.getFloatAttr(Type, Value));
}

bool ConstantFloatOp::classof(Operation *Op) {
  if (!ConstantOp::classof(Op))
    return false;

  mlir::Type T = Op->getResult(0).getType();
  return T.isa<FloatType>();
}

//===----------------------------------------------------------------------===//
//                                    LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::print(OpAsmPrinter &P) {
  P.printOptionalAttrDict((*this)->getAttrs());
  P << " " << getFrom() << " : " << getFrom().getType();
}

ParseResult LoadOp::parse(OpAsmParser &Parser, OperationState &Result) {
  OpAsmParser::UnresolvedOperand From;
  Type FromType;
  if (Parser.parseOptionalAttrDict(Result.attributes) ||
      Parser.parseOperand(From) || Parser.parseColon() ||
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

LogicalResult StoreOp::verify() {
  Type WhatType = getWhat().getType();
  PointerType WhereType = getWhere().getType().cast<PointerType>();

  if (WhereType.getElementType() == WhatType)
    return success();

  return emitError() << "type of stored value (" << WhatType
                     << ") doesn't match the pointer type (" << WhereType
                     << ")";
}

void StoreOp::print(OpAsmPrinter &P) {
  P.printOptionalAttrDict((*this)->getAttrs());
  P << " " << getWhat() << " -> " << getWhere() << " : "
    << getWhere().getType();
}

ParseResult StoreOp::parse(OpAsmParser &Parser, OperationState &Result) {
  OpAsmParser::UnresolvedOperand What, Where;
  Type WhereType;
  if (Parser.parseOptionalAttrDict(Result.attributes) ||
      Parser.parseOperand(What) || Parser.parseArrow() ||
      Parser.parseOperand(Where) || Parser.parseColon())
    return failure();

  Type WhatType = getPointee(WhereType);
  if (!WhatType || Parser.resolveOperand(What, WhatType, Result.operands) ||
      Parser.resolveOperand(Where, WhereType, Result.operands))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
//                               Cast operations
//===----------------------------------------------------------------------===//

bool BitcastOp::areCastCompatible(TypeRange Inputs, TypeRange Outputs) {
  assert(Inputs.size() == 1 && Outputs.size() == 1 &&
         "bitcast op expects one operand and result");
  Type From = Inputs.front(), To = Outputs.front();

  return (From.isIntOrFloat() && To.isIntOrFloat() &&
          From.getIntOrFloatBitWidth() == To.getIntOrFloatBitWidth()) ||
         (From.isa<air::PointerType>() && To.isa<air::PointerType>());
}

bool SExtOp::areCastCompatible(TypeRange Inputs, TypeRange Outputs) {
  assert(Inputs.size() == 1 && Outputs.size() == 1 &&
         "sext op expects one operand and result");
  Type From = Inputs.front(), To = Outputs.front();

  return From.isSignedInteger() && To.isa<IntegerType>() &&
         From.getIntOrFloatBitWidth() < To.getIntOrFloatBitWidth();
}

bool ZExtOp::areCastCompatible(TypeRange Inputs, TypeRange Outputs) {
  assert(Inputs.size() == 1 && Outputs.size() == 1 &&
         "zext op expects one operand and result");
  Type From = Inputs.front(), To = Outputs.front();

  return From.isUnsignedInteger() && To.isa<IntegerType>() &&
         From.getIntOrFloatBitWidth() < To.getIntOrFloatBitWidth();
}

bool TruncateOp::areCastCompatible(TypeRange Inputs, TypeRange Outputs) {
  assert(Inputs.size() == 1 && Outputs.size() == 1 &&
         "trunc op expects one operand and result");
  Type From = Inputs.front(), To = Outputs.front();

  return From.isIntOrFloat() && To.isIntOrFloat() &&
         From.getIntOrFloatBitWidth() > To.getIntOrFloatBitWidth();
}

bool CastToBaseOp::areCastCompatible(TypeRange Inputs, TypeRange Outputs) {
  assert(Inputs.size() == 1 && Outputs.size() == 1 &&
         "tobase op expects one operand and result");
  Type From = Inputs.front(), To = Outputs.front();

  if (!From.isa<air::PointerType>() || !To.isa<air::PointerType>())
    return false;

  Type FromPointee = From.cast<air::PointerType>().getElementType(),
       ToPointee = To.cast<air::PointerType>().getElementType();

  // TODO: actually check inheritance hierarchy
  return !(!FromPointee.isa<air::RecordRefType>() ||
           !ToPointee.isa<air::RecordRefType>());
}

//===----------------------------------------------------------------------===//
//                              Conditional branch
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands CondBranchOp::getSuccessorOperands(unsigned Index) {
  assert(Index < getNumSuccessors() && "invalid successor index");
  const MutableOperandRange Range =
      (Index == TrueIndex ? getTrueDestOperandsMutable()
                          : getFalseDestOperandsMutable());
  return SuccessorOperands{Range};
}

Block *CondBranchOp::getSuccessorForOperands(ArrayRef<Attribute> Operands) {
  if (IntegerAttr CondAttr =
          llvm::dyn_cast_or_null<IntegerAttr>(Operands.front()))
    return CondAttr.getValue().isOne() ? getTrueDest() : getFalseDest();
  return nullptr;
}

//===----------------------------------------------------------------------===//
//                        Record declaration/definition
//===----------------------------------------------------------------------===//

void RecordDeclOp::print(OpAsmPrinter &P) {
  const auto Name =
      getOperation()
          ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  P << " ";
  P.printSymbolName(Name);
}

ParseResult RecordDeclOp::parse(OpAsmParser &Parser, OperationState &Result) {
  // TODO: support decl parsing
  return failure();
}

RecordDefOp RecordDefOp::create(Location Loc, llvm::StringRef Name,
                                RecordType Record) {
  OpBuilder Builder(Loc->getContext());
  OperationState State(Loc, getOperationName());
  RecordDefOp::build(Builder, State, Name, Record);
  return cast<RecordDefOp>(Operation::create(State));
}

RecordDeclOp RecordDeclOp::create(Location Loc, llvm::StringRef Name) {
  OpBuilder Builder(Loc->getContext());
  OperationState State(Loc, getOperationName());
  RecordDeclOp::build(Builder, State, Name);
  return cast<RecordDeclOp>(Operation::create(State));
}

void RecordDefOp::print(OpAsmPrinter &P) {
  const auto Name =
      getOperation()
          ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  P << " ";
  P.printSymbolName(getSymNameAttr());
  P << " : ";
  P.printType(getType());
}

ParseResult RecordDefOp::parse(OpAsmParser &Parser, OperationState &Result) {
  // TODO: implement
  return failure();
}

//===----------------------------------------------------------------------===//
//                                    sizeof
//===----------------------------------------------------------------------===//

void SizeOfOp::print(OpAsmPrinter &P) {
  P << " ";
  P.printType(getType());
  P << " : ";
  P.printType(getRes().getType());
}

ParseResult SizeOfOp::parse(OpAsmParser &Parser, OperationState &Result) {
  // TODO: implement
  return failure();
}

#define GET_OP_CLASSES
#include "tau/AIR/AirOps.cpp.inc"
