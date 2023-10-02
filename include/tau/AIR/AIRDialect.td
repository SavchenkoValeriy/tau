//===- AirDialect.td - AIR dialect -------------------------*- tablegen -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
// TBD
//
//===----------------------------------------------------------------------===//

#ifndef AIRDIALECT_H
#define AIRDIALECT_H

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
//                              Dialect definition
//===----------------------------------------------------------------------===//

def Air_Dialect : Dialect {
    let name = "air";
    let summary = "Analysis-purposed MLIR dialect.";
    let description = [{
    Static analysis-purposed IR for Tau.
    }];
    let cppNamespace = "tau::air";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
    private:
      void registerTypes();

    public:
    }];
}

//===----------------------------------------------------------------------===//
//                          Base operation definition
//===----------------------------------------------------------------------===//

class Air_Op<string mnemonic, list<Trait> traits = []> :
    Op<Air_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
//                                Air own types
//===----------------------------------------------------------------------===//

class Air_TypeDef<string name, string typeMnemonic,
                  list<Trait> traits = [],
                  string baseCppClass = "::mlir::Type">
    : TypeDef<Air_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Air_PointerType : Air_TypeDef<"Pointer", "ptr"> {
  let parameters = (ins "mlir::Type":$elementType);
  let assemblyFormat = "$elementType";
  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$type), [{
      return $_get(type.getContext(), type);
    }]>
  ];
}

def Air_RecordRefType : Air_TypeDef<"RecordRef", "recref"> {
  let parameters = (ins "mlir::FlatSymbolRefAttr":$name);
  let assemblyFormat = "$name";
  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::MLIRContext *":$context, "llvm::StringRef":$name), [{
      return $_get(context, SymbolRefAttr::get(context, name));
    }]>
  ];
}

def Air_RecordType : Air_TypeDef<"Record", "rec"> {
  let parameters = (ins "llvm::ArrayRef<mlir::Type>":$Bases,
                        "llvm::ArrayRef<RecordField>":$Fields);
  let hasCustomAssemblyFormat = 1;
  let genStorageClass = 0;

  let extraClassDeclaration = [{
    RecordField getFieldByIndex(unsigned Index) const;
    mlir::Type getTypeByIndex(unsigned Index) const {
      return getFieldByIndex(Index).Type;
    }
    llvm::StringRef getFieldNameByIndex(unsigned Index) const {
      return getFieldByIndex(Index).Name;
    }

    RecordField getFieldByName(llvm::StringRef Name) const;
    mlir::Type getTypeByName(llvm::StringRef Name) const {
      return getFieldByName(Name).Type;
    }
  }];
}

def Air_VoidType : Air_TypeDef<"Void", "void"> {
}

//===----------------------------------------------------------------------===//
//                             Air type predicates
//===----------------------------------------------------------------------===//

def Air_Type : DialectType<Air_Dialect,
                           CPred<"tau::air::isCompatibleType($_self)">,
                           "Air-compatible type">;

// Type constraint accepting any Air pointer type.
def Air_AnyPointer : Type<CPred<"$_self.isa<tau::air::PointerType>()">,
                          "Air pointer type">;

// Type constraint accepting Air pointer type with an additional constraint
// on the element type.
class Air_PointerTo<Type pointee> : Type<
  And<[Air_AnyPointer.predicate,
       SubstLeaves<
         "$_self",
         "$_self.cast<tau::air::PointerType>().getElementType()",
         pointee.predicate>]>,
  "Air pointer to " # pointee.summary>;

def Air_AnyRecord : Type<CPred<"$_self.isa<tau::air::RecordType>()">,
                         "Air record type">;

def Air_AnyRecordRef : Type<CPred<"$_self.isa<tau::air::RecordRefType>()">,
                            "Air record reference type">;

//===----------------------------------------------------------------------===//
//                                Air attributes
//===----------------------------------------------------------------------===//

def Air_StateChangeAttr : DialectAttr<
    Air_Dialect,
    CPred<"$_self.isa<tau::air::StateChangeAttr>()">,
    "Checker annotation for state change"> {
}

def Air_StateTransferAttr : DialectAttr<
    Air_Dialect,
    CPred<"$_self.isa<tau::air::StateTransferAttr>()">,
    "Checker annotation for state tarnsfer"> {
}

#endif /* AIRDIALECT_H */
