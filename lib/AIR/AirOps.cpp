#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirTypes.h"

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

#define GET_OP_CLASSES
#include "tau/AIR/AirOps.cpp.inc"
