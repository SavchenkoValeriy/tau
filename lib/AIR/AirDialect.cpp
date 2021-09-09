#include "tau/AIR/AirDialect.h"
#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirTypes.h"

#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace tau::air;

#include "tau/AIR/AirOpsDialect.cpp.inc"

void AirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tau/AIR/AirOps.cpp.inc"
      >();
  addTypes<PointerType>();
}

/// Parse a type registered to this dialect.
Type AirDialect::parseType(DialectAsmParser &Parser) const {
  Type Result;
  // TODO: implement parsing for air.ptr.
  if (Parser.parseType(Result))
    return {};

  return Result;
}

/// Print a type registered to this dialect.
void AirDialect::printType(Type TypeToPrint, DialectAsmPrinter &OS) const {
  if (auto PtrType = TypeToPrint.dyn_cast<PointerType>()) {
    OS << "ptr";
    OS << "<";
    OS.printType(PtrType.getElementType());
    OS << ">";
  } else {
    OS.printType(TypeToPrint);
  }
}
