#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirOps.h"

using namespace tau::air;

#include "tau/AIR/AirOpsDialect.cpp.inc"

void AirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tau/AIR/AirOps.cpp.inc"
      >();
}
