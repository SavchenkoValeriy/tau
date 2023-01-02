#include "tau/AIR/AirAttrs.h"

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/StringRef.h>

using namespace tau::air;
using namespace llvm;

//===----------------------------------------------------------------------===//
//                               State attributes
//===----------------------------------------------------------------------===//

StateChangeAttr StateChangeAttr::get(mlir::MLIRContext *Context,
                                     llvm::StringRef CheckerID,
                                     unsigned int OperandIdx, StateID From,
                                     StateID To) {
  return Base::get(Context, CheckerID, OperandIdx, From, To);
}

StateChangeAttr StateChangeAttr::get(mlir::MLIRContext *Context,
                                     llvm::StringRef CheckerID,
                                     unsigned int OperandIdx, StateID To) {
  return Base::get(Context, CheckerID, OperandIdx, 0, To);
}

StringRef StateChangeAttr::getCheckerID() const {
  return std::get<0>(getImpl()->Key);
}

unsigned StateChangeAttr::getOperandIdx() const {
  return std::get<1>(getImpl()->Key);
}

Optional<StateID> StateChangeAttr::getFromState() const {
  if (StateID Result = StateID::fromRaw(std::get<2>(getImpl()->Key)))
    return Result;
  return None;
}

StateID StateChangeAttr::getToState() const {
  return StateID::fromRaw(std::get<3>(getImpl()->Key));
}

StateTransferAttr StateTransferAttr::get(mlir::MLIRContext *Context,
                                         llvm::StringRef CheckerID,
                                         unsigned FromOperandIdx,
                                         unsigned ToOperandIdx,
                                         StateID LimitedTo) {
  return Base::get(Context, CheckerID, FromOperandIdx, ToOperandIdx, LimitedTo);
}

StateTransferAttr StateTransferAttr::get(mlir::MLIRContext *Context,
                                         llvm::StringRef CheckerID,
                                         unsigned FromOperandIdx,
                                         unsigned ToOperandIdx) {
  return Base::get(Context, CheckerID, FromOperandIdx, ToOperandIdx,
                   StateID::fromRaw(0));
}

StringRef StateTransferAttr::getCheckerID() const {
  return std::get<0>(getImpl()->Key);
}

unsigned StateTransferAttr::getFromOperandIdx() const {
  return std::get<1>(getImpl()->Key);
}

unsigned StateTransferAttr::getToOperandIdx() const {
  return std::get<2>(getImpl()->Key);
}

Optional<StateID> StateTransferAttr::getLimitingState() const {
  if (StateID Result = StateID::fromRaw(std::get<3>(getImpl()->Key)))
    return Result;
  return None;
}
