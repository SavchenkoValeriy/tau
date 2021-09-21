#include "tau/Core/State.h"

#include <mlir/IR/Operation.h>

using namespace llvm;
using namespace mlir;

ArrayRef<Attribute> tau::core::getStateAttributes(Operation *Op) {
  auto Attr = Op->getAttrOfType<ArrayAttr>(StateAttrID);
  return Attr ? Attr.getValue() : ArrayRef<Attribute>();
}
