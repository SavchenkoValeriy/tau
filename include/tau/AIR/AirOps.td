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

def Air_NegIOp : Air_Op<"negi", [NoSideEffect,
                                 SameOperandsAndResultType]>,
                Arguments<(ins AnyInteger:$operand)> {
    let results = (outs AnyInteger);
    let summary = "integer negation";
    let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.negi` ssa-use `:` type
    ```

    The `neg` operation takes one operand and returns one result, operand and
    result types are the same. This type should be an integer scalar type.
    It has no standard attributes.

    Example:

    ```mlir
    %a = negi %b : i32
    ```
  }];
}

//===----------------------------------------------------------------------===//
//                              Memory operations
//===----------------------------------------------------------------------===//

def Air_AllocaOp : Air_Op<"alloca"> {
    let arguments = (ins Optional<AnyUnsignedInteger>:$length);
    let results = (outs Air_AnyPointer:$res);
    let summary = "alloca operation";
    let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.alloca` (ssa-use : type)? `:` type
    ```

    The `alloca` operation represents all allocation on stack. It takes
    one optional `length` argument for array allocation. The result type
    is a pointer type. Operation has no standard attributes.

    Example:

    ```mlir
    %a = alloca : !air.ptr<i32>

    %b = alloca %length : !air.ptr<f64>
    ```
    }];
    let assemblyFormat = "($length^ `:` type($length))? attr-dict `:` type($res)";
    let verifier = ?;
}

def Air_StoreOp : Air_Op<"store"> {
    let arguments = (ins AnyType:$what, Air_AnyPointer:$where);
    let summary = "store operation";
    let description = [{
    Syntax:

    ```
    operation ::= `air.store` ssa-use `:` type `->` ssa-use `:` type
    ```

    The `store` operation represents writing value into a block of
    memory pointed to by the given location. It takes two arguments
    that should differ only in pointer type. It has no results and
    standard attributes.
    }];
    let assemblyFormat = "attr-dict $what `:` type($what) `->` $where `:` type($where)";
    let verifier = [{ return ::verify(*this); }];
}

#endif /* AIROPS_H */
