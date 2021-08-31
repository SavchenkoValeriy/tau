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
}

//===----------------------------------------------------------------------===//
//                          Base operation definition
//===----------------------------------------------------------------------===//

class Air_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Air_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
//                                Air own types
//===----------------------------------------------------------------------===//

def Air_Type : DialectType<Air_Dialect,
                           CPred<"tau::air::isCompatibleType($_self)">,
                           "Air-compatible type">;

// Type constraint accepting any Air pointer type.
def Air_AnyPointer : Type<CPred<"$_self.isa<tau::air::AirPointerType>()">,
                          "Air pointer type">;

// Type constraint accepting Air pointer type with an additional constraint
// on the element type.
class Air_PointerTo<Type pointee> : Type<
  And<[Air_AnyPointer.predicate,
       SubstLeaves<
         "$_self",
         "$_self.cast<tau::air::AirPointerType>().getElementType()",
         pointee.predicate>]>,
  "Air pointer to " # pointee.summary>;

#endif /* AIRDIALECT_H */
