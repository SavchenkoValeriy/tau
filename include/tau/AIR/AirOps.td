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

//===----------------------------------------------------------------------===//
//                               Unary operations
//===----------------------------------------------------------------------===//

class UnaryIntOp<string mnemonic, list<OpTrait> traits = []> :
      Air_Op<mnemonic, !listconcat(traits, [NoSideEffect,
                                            SameOperandsAndResultType])> {
  let arguments = (ins AnyInteger:$operand);
  let results = (outs AnyInteger:$res);
  let assemblyFormat = "attr-dict $operand `:` type($res)";
}

def Air_NotOp : UnaryIntOp<"not"> {
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

def Air_NegIOp : UnaryIntOp<"negi"> {
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
//                              Binary operations
//===----------------------------------------------------------------------===//

class BinaryIntOp<string mnemonic, list<OpTrait> traits = []> :
      Air_Op<mnemonic, !listconcat(traits, [NoSideEffect,
                                            SameOperandsAndResultType])> {
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$res);
  let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($res)";
}

def Air_AddIOp : BinaryIntOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.addi` ssa-use `,` ssa-use `:` type
    ```

    The `addi` operation takes two operands and returns one result, each of
    these is required to be the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = air.addi %b, %c : si64
    ```
  }];
}

def Air_SubIOp : BinaryIntOp<"subi"> {
  let summary = "integer subtraction operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.subi` ssa-use `,` ssa-use `:` type
    ```

    The `subi` operation takes two operands and returns one result, each of
    these is required to be the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = air.subi %b, %c : si64
    ```
  }];
}

def Air_MulIOp : BinaryIntOp<"muli", [Commutative]> {
  let summary = "integer multiplication operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.muli` ssa-use `,` ssa-use `:` type
    ```

    The `muli` operation takes two operands and returns one result, each of
    these is required to be the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = air.muli %b, %c : si64
    ```
  }];
}

def Air_XOrOp : BinaryIntOp<"xor", [Commutative]> {
  let summary = "integer binary xor";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.xor` ssa-use `,` ssa-use `:` type
    ```

    The `xor` operation takes two operands and returns one result, each of
    these is required to be the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = air.xor %b, %c : si64
    ```
  }];
}

def Air_OrOp : BinaryIntOp<"or", [Commutative]> {
  let summary = "integer binary or";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.or` ssa-use `,` ssa-use `:` type
    ```

    The `or` operation takes two operands and returns one result, each of
    these is required to be the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = air.or %b, %c : si64
    ```
  }];
}

def Air_AndOp : BinaryIntOp<"and", [Commutative]> {
  let summary = "integer binary and";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.and` ssa-use `,` ssa-use `:` type
    ```

    The `and` operation takes two operands and returns one result, each of
    these is required to be the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = air.and %b, %c : si64
    ```
  }];
}

def Air_ShiftLeftOp : BinaryIntOp<"shl"> {
  let summary = "integer left-shift";
  let description = [{
    The shift_left operation shifts an integer value to the left by a variable
    amount. The low order bits are filled with zeros.

    Example:

    ```mlir
    %1 = constant 5 : i8        // %1 is 0b00000101
    %2 = constant 3 : i8
    %3 = air.shl %1, %2 : i8    // %3 is 0b00101000
    ```
  }];
}

def Air_ArithmeticShiftRightOp : BinaryIntOp<"ashr"> {
  let summary = "arithmetic (signed) integer right-shift";
  let description = [{
    The shift_right_signed operation shifts an integer value to the right by
    a variable amount. The integer is interpreted as signed. The high order
    bits in the output are filled with copies of the most-significant bit
    of the shifted value (which means that the sign of the value is preserved).

    Example:

    ```mlir
    %1 = air.constant 160 : si8 // %1 is 0b10100000
    %2 = air.constant 3 : si8
    %3 = air.ashr %1, %2 : si8  // %3 is 0b11110100
    %4 = air.constant 96 : si8  // %4 is 0b01100000
    %5 = air.ashr %4, %2 : si8  // %5 is 0b00001100
    ```
  }];
}

def Air_LogicalShiftRightOp : BinaryIntOp<"lshr"> {
  let summary = "logical (unsigned) integer right-shift";
  let description = [{
    The shift_right_unsigned operation shifts an integer value to the right by
    a variable amount. The integer is interpreted as unsigned. The high order
    bits are always filled with zeros.

    Example:

    ```mlir
    %1 = air.constant 160 : ui8  // %1 is 0b10100000
    %2 = air.constant 3 : ui8
    %3 = air.lshr %1, %2 : ui8   // %3 is 0b00010100
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

def Air_LoadOp : Air_Op<"load"> {
  let arguments = (ins Air_AnyPointer:$from);
  let results = (outs AnyType:$res);
  let summary = "load operation";
  let description = [{
    Syntax:

    ```
    operation ::= `air.load` ssa-use `:` type
    ```

    The `load` operation represents reading value from a block of
    memory pointed to by the given location. It takes one argument
    of a pointer type, and one result of the pointee type. The operation
    has no standard attributes.
    }];
  let builders = [
      OpBuilder<(ins "mlir::Value":$from),
      [{
        build($_builder, $_state, from.getType().cast<AirPointerType>().getElementType(), from);
      }]>];
  let verifier = ?;
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseLoad(parser, result); }];
}

//===----------------------------------------------------------------------===//
//                                   Constant
//===----------------------------------------------------------------------===//

def Air_ConstantOp : Air_Op<"constant", [NoSideEffect, ConstantLike]> {
  let summary = "constant";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.constant` attribute-value `:` type
    ```

    The `constant` operation produces an SSA value equal to some constant
    specified by an attribute.
    Example:

    ```mlir
    // Integer constant
    %1 = "air.constant" 42 : i32
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$res);

  let builders = [
    OpBuilder<(ins "mlir::Attribute":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>,
    OpBuilder<(ins "mlir::Attribute":$value, "mlir::Type":$type),
    [{ build($_builder, $_state, type, value); }]>,
  ];

  let extraClassDeclaration = [{
    mlir::Attribute getValue() { return (*this)->getAttr("value"); }
  }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseConstantOp(parser, result); }];
}


#endif /* AIROPS_H */
