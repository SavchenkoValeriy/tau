#include "tau/AIR/AirDialect.h"

#include "tau/AIR/AirAttrs.h"
#include "tau/AIR/AirOps.h"
#include "tau/AIR/AirTypes.h"

#include <mlir/IR/Attributes.h>
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
  addAttributes<StateChangeAttr, StateTransferAttr>();
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

Attribute AirDialect::parseAttribute(DialectAsmParser &Parser, Type) const {
  Attribute Result;
  // TODO: implement parsing for air attributes
  if (Parser.parseAttribute(Result))
    return {};

  return Result;
}

void AirDialect::printAttribute(Attribute Attr, DialectAsmPrinter &OS) const {
  if (auto Change = Attr.dyn_cast<StateChangeAttr>()) {
    OS << "change";
    OS << '<';
    {
      OS << '"' << Change.getCheckerID() << '"';
      OS << ' ' << Change.getOperandIdx();
      OS << '(';
      {
        if (auto From = Change.getFromState())
          OS << From;
        OS << "->";
        OS << Change.getToState();
      }
      OS << ')';
    }
    OS << '>';
  } else if (auto Transfer = Attr.dyn_cast<StateTransferAttr>()) {
    OS << "transfer";
    OS << '<';
    {
      OS << '"' << Transfer.getCheckerID() << '"';
      OS << ' ';
      OS << Transfer.getFromOperandIdx();
      OS << " -> ";
      OS << Transfer.getToOperandIdx();
      if (auto LimitedTo = Transfer.getLimitingState()) {
        OS << '(';
        OS << *LimitedTo;
        OS << ')';
      }
    }
    OS << '>';
  } else {
    OS.printAttribute(Attr);
  }
}
