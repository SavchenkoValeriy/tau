//===- AirOps.td - Air operations --------------------------*- tablegen -*-===//
//
// Part of the Tau Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
//  TBD
//
//===----------------------------------------------------------------------===//

#ifndef AIROPS_H
#define AIROPS_H

include "AirDialect.td"
include "mlir/IR/OpBase.td"

def Air_NotOp : Air_Op<"not", [NoSideEffect,
                               SameOperandsAndResultType]>,
                Arguments<(ins AnyInteger:$operand)> {
    let results = (outs AnyInteger);
    let summary = "integer binary not";
    let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.not` ssa-use `:` type
    ```

    The `not` operation takes one operand and returns one result, operand and
    result types are the same. This type should be an integer scalar type.
    It has no standard attributes.

    Example:

    ```mlir
    %a = not %b : i64
    ```
  }];
}

#endif /* AIROPS_H */
