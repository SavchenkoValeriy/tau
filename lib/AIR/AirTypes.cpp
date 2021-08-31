#include "tau/AIR/AirTypes.h"

using namespace tau::air;
using namespace mlir;

//===----------------------------------------------------------------------===//
//                                 Pointer type
//===----------------------------------------------------------------------===//

AirPointerType AirPointerType::get(Type Pointee) {
  assert(Pointee && "expected non-null subtype");
  return Base::get(Pointee.getContext(), Pointee);
}

Type AirPointerType::getElementType() const { return getImpl()->PointeeType; }
