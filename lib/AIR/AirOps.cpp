#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirTypes.h"

#include <mlir/IR/OpImplementation.h>

#define GET_OP_CLASSES
#include "tau/AIR/AirOps.cpp.inc"

using namespace mlir;
using namespace tau::air;
