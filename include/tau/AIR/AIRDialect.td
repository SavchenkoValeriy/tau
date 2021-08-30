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

#endif /* AIRDIALECT_H */
