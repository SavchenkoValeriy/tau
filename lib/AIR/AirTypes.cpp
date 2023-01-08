#include "tau/AIR/AirTypes.h"

#include "tau/AIR/AirDialect.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace tau::air;
using namespace mlir;

void AirDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tau/AIR/AirOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
//                                 Record type
//===----------------------------------------------------------------------===//

ArrayRef<RecordField> RecordType::getFields() const {
  return getImpl()->Fields;
}

RecordField RecordType::getFieldByIndex(unsigned Index) const {
  assert(Index < getImpl()->Fields.size() && "accessing unknown field");
  return getImpl()->Fields[Index];
}

RecordField RecordType::getFieldByName(StringRef Name) const {
  const auto Fields = getImpl()->Fields;
  const auto *It = llvm::find_if(
      Fields, [Name](const RecordField &Field) { return Field.Name == Name; });
  assert(It != Fields.end() && "accessing unknown field");
  return *It;
}

Type tau::air::RecordType::parse(mlir::AsmParser &P) {
  SmallVector<RecordField, 4> Fields;
  SmallVector<std::string> Names{""};

  if (P.parseCommaSeparatedList(mlir::AsmParser::Delimiter::LessGreater, [&]() {
        StringRef Name = Names.back();
        Type FieldType;
        if (P.parseKeyword(&Name) || P.parseColonType(FieldType))
          return ParseResult::failure();
        Names.emplace_back("");
        Fields.push_back({Name, FieldType});
        return ParseResult::success();
      }))
    return Type();

  return get(P.getContext(), Fields);
}

void tau::air::RecordType::print(mlir::AsmPrinter &P) const {
  P << "<";
  llvm::interleaveComma(getFields(), P, [&](const RecordField &Field) {
    P << Field.Name << " : " << Field.Type;
  });
  P << ">";
}

#define GET_TYPEDEF_CLASSES
#include "tau/AIR/AirOpsTypes.cpp.inc"
