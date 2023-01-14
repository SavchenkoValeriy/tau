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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
//                               Unary operations
//===----------------------------------------------------------------------===//

class UnaryIntOp<string mnemonic, list<Trait> traits = []> :
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

class BinaryIntOp<string mnemonic, list<Trait> traits = []> :
      Air_Op<mnemonic, !listconcat(traits, [NoSideEffect,
                                            SameOperandsAndResultType])> {
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$res);
  let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($res)";
}

class BinarySIntOp<string mnemonic, list<Trait> traits = []> :
      BinaryIntOp<mnemonic, traits> {
  let arguments = (ins AnySignedInteger:$lhs, AnySignedInteger:$rhs);
}

class BinaryUIntOp<string mnemonic, list<Trait> traits = []> :
      BinaryIntOp<mnemonic, traits> {
  let arguments = (ins AnyUnsignedInteger:$lhs, AnyUnsignedInteger:$rhs);
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

def Air_SignedDivIOp : BinarySIntOp<"sdiv"> {
  let summary = "2's complement signed division";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.sdiv` ssa-use `,` ssa-use `:` type
    ```

    The `sdiv` operation takes two operands and returns one result, each of
    these is required to be the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = air.sdiv %b, %c : si64
    ```
  }];
}

def Air_UnsignedDivIOp : BinaryUIntOp<"udiv"> {
  let summary = "unsigned integer division";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.udiv` ssa-use `,` ssa-use `:` type
    ```

    The `udiv` operation takes two operands and returns one result, each of
    these is required to be the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = air.udiv %b, %c : si64
    ```
  }];
}

def Air_SignedRemIOp : BinarySIntOp<"srem"> {
  let summary = "2's complement signed remainder";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.srem` ssa-use `,` ssa-use `:` type
    ```

    The `srem` operation takes two operands and returns one result, each of
    these is required to be the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = air.srem %b, %c : si64
    ```
  }];
}

def Air_UnsignedRemIOp : BinaryUIntOp<"urem"> {
  let summary = "unsigned integer remainder";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.urem` ssa-use `,` ssa-use `:` type
    ```

    The `urem` operation takes two operands and returns one result, each of
    these is required to be the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = air.urem %b, %c : si64
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
    The `shl` operation shifts an integer value to the left by a variable
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
    The `ashr` operation shifts an integer value to the right by
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
    The `lshr` operation shifts an integer value to the right by
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
//                        Integer comparison operations
//===----------------------------------------------------------------------===//

class ComparisonOp<string mnemonic, list<Trait> traits = []> :
      Air_Op<mnemonic, !listconcat(traits, [NoSideEffect,
                                            SameTypeOperands])> {
  let results = (outs UI1:$res);
  let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($lhs)";
}

class ComparisonIntOp<string mnemonic, list<Trait> traits = []> :
      ComparisonOp<mnemonic, traits> {
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
}

class ComparisonSIntOp<string mnemonic, list<Trait> traits = []> :
      ComparisonOp<mnemonic, traits> {
  let arguments = (ins AnySignedInteger:$lhs, AnySignedInteger:$rhs);
}

class ComparisonUIntOp<string mnemonic, list<Trait> traits = []> :
      ComparisonOp<mnemonic, traits> {
  let arguments = (ins AnyUnsignedInteger:$lhs, AnyUnsignedInteger:$rhs);
}

class ComparisonFloatOp<string mnemonic, list<Trait> traits = []> :
      ComparisonOp<mnemonic, traits> {
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
}

def Air_LessThanSIOp : ComparisonSIntOp<"slt"> {
  let summary = "signed less than";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.slt` ssa-use, ssa-use `:` type
    ```

    The `slt` operation compares its signed integer arguments and returns 1
    if `lhs` is less than `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.slt %0, %1 : si32
    ```
  }];
}

def Air_LessThanUIOp : ComparisonUIntOp<"ult"> {
  let summary = "unsigned less than";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.ult` ssa-use, ssa-use `:` type
    ```

    The `ult` operation compares its unsigned integer arguments and returns 1
    if `lhs` is less than `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.ult %0, %1 : si32
    ```
  }];
}

def Air_LessThanFOp : ComparisonFloatOp<"ltf"> {
  let summary = "floating point less than";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.ltf` ssa-use, ssa-use `:` type
    ```

    The `ltf` operation compares its floating point arguments and returns 1
    if `lhs` is less than `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.ltf %0, %1 : si32
    ```
  }];
}

def Air_LessThanOrEqualSIOp : ComparisonSIntOp<"sle"> {
  let summary = "signed less than or equal";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.sle` ssa-use, ssa-use `:` type
    ```

    The `sle` operation compares its signed integer arguments and returns 1
    if `lhs` is less than or equal `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.sle %0, %1 : si32
    ```
  }];
}

def Air_LessThanOrEqualUIOp : ComparisonUIntOp<"ule"> {
  let summary = "unsigned less than or equal";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.ule` ssa-use, ssa-use `:` type
    ```

    The `ule` operation compares its unsigned integer arguments and returns 1
    if `lhs` is less than or equal `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The resule type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.ule %0, %1 : si32
    ```
  }];
}

def Air_LessThanOrEqualFOp : ComparisonFloatOp<"lef"> {
  let summary = "floating point less than or equal";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.lef` ssa-use, ssa-use `:` type
    ```

    The `lef` operation compares its signed integer arguments and returns 1
    if `lhs` is less than or equal `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.lef %0, %1 : si32
    ```
  }];
}

def Air_GreaterThanSIOp : ComparisonSIntOp<"sgt"> {
  let summary = "signed greater than";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.sgt` ssa-use, ssa-use `:` type
    ```

    The `sgt` operation compares its signed integer arguments and returns 1
    if `lhs` is greater than `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.sgt %0, %1 : si32
    ```
  }];
}

def Air_GreaterThanUIOp : ComparisonUIntOp<"ugt"> {
  let summary = "unsigned greater than";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.ugt` ssa-use, ssa-use `:` type
    ```

    The `ugt` operation compares its unsigned integer arguments and returns 1
    if `lhs` is greater than `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.ugt %0, %1 : si32
    ```
  }];
}

def Air_GreaterThanFOp : ComparisonFloatOp<"gtf"> {
  let summary = "floating point greater than";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.gtf` ssa-use, ssa-use `:` type
    ```

    The `gtf` operation compares its floating point arguments and returns 1
    if `lhs` is greater than `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.gtf %0, %1 : si32
    ```
  }];
}

def Air_GreaterThanOrEqualSIOp : ComparisonSIntOp<"sge"> {
  let summary = "signed greater than or equal";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.sge` ssa-use, ssa-use `:` type
    ```

    The `sge` operation compares its signed integer arguments and returns 1
    if `lhs` is greater than or equal `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.sge %0, %1 : si32
    ```
  }];
}

def Air_GreaterThanOrEqualUIOp : ComparisonUIntOp<"uge"> {
  let summary = "unsigned greater than or equal";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.uge` ssa-use, ssa-use `:` type
    ```

    The `uge` operation compares its unsigned integer arguments and returns 1
    if `lhs` is greater than or equal `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.uge %0, %1 : si32
    ```
  }];
}

def Air_GreaterThanOrEqualFOp : ComparisonFloatOp<"gef"> {
  let summary = "floating point greater than or equal";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.gef` ssa-use, ssa-use `:` type
    ```

    The `gef` operation compares its signed integer arguments and returns 1
    if `lhs` is greater than or equal `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.gef %0, %1 : si32
    ```
  }];
}

def Air_EqualIOp : ComparisonIntOp<"eqi", [Commutative]> {
  let summary = "integer equal";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.eqi` ssa-use, ssa-use `:` type
    ```

    The `eqi` operation compares its floating point arguments and returns 1
    if `lhs` is equal `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.eqi %0, %1 : si32
    ```
  }];
}

def Air_EqualFOp : ComparisonFloatOp<"eqf", [Commutative]> {
  let summary = "floating point equal";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.eqf` ssa-use, ssa-use `:` type
    ```

    The `eqf` operation compares its floating point arguments and returns 1
    if `lhs` is equal `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.eqf %0, %1 : si32
    ```
  }];
}

def Air_NotEqualIOp : ComparisonIntOp<"nei", [Commutative]> {
  let summary = "integer not equal";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.nei` ssa-use, ssa-use `:` type
    ```

    The `nei` operation compares its floating point arguments and returns 1
    if `lhs` is not equal `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.nei %0, %1 : si32
    ```
  }];
}

def Air_NotEqualFOp : ComparisonFloatOp<"nef", [Commutative]> {
  let summary = "floating point not equal";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.nef` ssa-use, ssa-use `:` type
    ```

    The `nef` operation compares its floating point arguments and returns 1
    if `lhs` is not equal `rhs`, and 0 otherwise. Operands are required to be
    of the same type. The result type is boolean. Operation has no standard
    attributes.

    Example:

    ```mlir
    %a = air.nef %0, %1 : si32
    ```
  }];
}

//===----------------------------------------------------------------------===//
//                              Conditional branch
//===----------------------------------------------------------------------===//

def CondBranchOp : Air_Op<"cond_br",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     NoSideEffect, Terminator]> {
  let summary = "conditional branch operation";
  let description = [{
    The `cond_br` terminator operation represents a conditional branch on a
    boolean (1-bit integer) value. If the bit is set, then the first destination
    is jumped to; if it is false, the second destination is chosen. The count
    and types of operands must align with the arguments in the corresponding
    target blocks.

    The MLIR conditional branch operation is not allowed to target the entry
    block for a region. The two destinations of the conditional branch operation
    are allowed to be the same.

    The following example illustrates a function with a conditional branch
    operation that targets the same block.

    Example:

    ```mlir
    func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
      // Both targets are the same, operands differ
      cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

    ^bb1(%x : i32) :
      return %x : i32
    }
    ```
  }];

  let arguments = (ins UI1:$condition,
                       Variadic<AnyType>:$trueDestOperands,
                       Variadic<AnyType>:$falseDestOperands);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

  let builders = [
    OpBuilder<(ins "mlir::Value":$condition, "mlir::Block *":$trueDest,
      "mlir::ValueRange":$trueOperands, "mlir::Block *":$falseDest,
      "mlir::ValueRange":$falseOperands), [{
      build($_builder, $_state, condition, trueOperands, falseOperands, trueDest,
            falseDest);
    }]>,
    OpBuilder<(ins "mlir::Value":$condition, "mlir::Block *":$trueDest,
      "mlir::Block *":$falseDest, CArg<"mlir::ValueRange", "{}">:$falseOperands), [{
      build($_builder, $_state, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
    }]>];

  // CondBranchOp is fully verified by traits.
  let hasVerifier = 0;

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { TrueIndex = 0, FalseIndex = 1 };

    // The condition operand is the first operand in the list.
    mlir::Value getCondition() { return getOperand(0); }

    /// Return the destination if the condition is true.
    mlir::Block *getTrueDest() {
      return getSuccessor(TrueIndex);
    }

    /// Return the destination if the condition is false.
    mlir::Block *getFalseDest() {
      return getSuccessor(FalseIndex);
    }

    // Accessors for operands to the 'true' destination.
    mlir::Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, mlir::Value value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    operand_range getTrueOperands() { return trueDestOperands(); }

    unsigned getNumTrueOperands()  { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      trueDestOperandsMutable().erase(index);
    }

    // Accessors for operands to the 'false' destination.
    mlir::Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, mlir::Value value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_range getFalseOperands() { return falseDestOperands(); }

    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      falseDestOperandsMutable().erase(index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  let assemblyFormat = [{
    $condition `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
//                           Integer cast operations
//===----------------------------------------------------------------------===//

class AirCastOp<string mnemonic, list<Trait> traits = []> :
    Air_Op<mnemonic, traits # [
      NoSideEffect, SameOperandsAndResultShape,
      DeclareOpInterfaceMethods<CastOpInterface>]> {
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$dest);
  let assemblyFormat = "attr-dict $source `:` type($source) `to` type($dest)";
}

def Air_BitcastOp : AirCastOp<"bitcast"> {
  let summary = "bitcast operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.bitcast` ssa-use : type `to` type
    ```

    Bitcast an integer or floating point value to an integer or floating point
    value of equal bit width. When operating on vectors, casts elementwise.

    Example:

    ```mlir
    %b = air.bitcast %a ui32 to si32
    ```
  }];
}

def Air_SExtOp : AirCastOp<"sext"> {
  let summary = "signed extend operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.sext` ssa-use : type `to` type
    ```

    The integer sign extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with copies
    of the most-significant bit of the input.

    Example:

    ```mlir
    %b = air.sext %a si32 to si64
    ```
  }];
}

def Air_ZExtOp : AirCastOp<"zext"> {
  let summary = "zero extend operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.zext` ssa-use : type `to` type
    ```

    The integer zero extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with zeros.

    Example:

    ```mlir
    %b = air.zext %a ui32 to ui64
    ```
  }];
}

def Air_TruncateOp : AirCastOp<"trunc"> {
  let summary = "integer truncation operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.zext` ssa-use : type `to` type
    ```

    The integer truncation operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be smaller than the input bit-width (N < M).
    The top-most (N - M) bits of the input are discarded.

    Example:

    ```mlir
      %1 = air.constant 21 : ui5           // %1 is 0b10101
      %2 = air.trunc %1 : ui5 to ui4       // %2 is 0b0101
      %3 = air.trunc %1 : ui5 to ui3       // %3 is 0b101
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
    %a = alloca : !air<ptr i32>

    %b = alloca %length : !air<ptr f64>
    ```
    }];
    let assemblyFormat = "($length^ `:` type($length))? attr-dict `:` type($res)";
    let hasVerifier = 0;
}

def Air_StoreOp : Air_Op<"store"> {
  let arguments = (ins AnyType:$what, Air_AnyPointer:$where);
  let summary = "store operation";
  let description = [{
    Syntax:

    ```
    operation ::= `air.store` ssa-use `->` ssa-use `:` type
    ```

    The `store` operation represents writing value into a block of
    memory pointed to by the given location. It takes two arguments
    that should differ only in pointer type. It has no results and
    standard attributes.
    }];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    mlir::Value getValue() {
      return getOperand(0);
    }

    mlir::Value getAddress() {
      return getOperand(1);
    }
  }];
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
        build($_builder, $_state, from.getType().cast<PointerType>().getElementType(), from);
      }]>];
  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    mlir::Value getAddress() {
      return from();
    }
  }];
}

def Air_RefOp : Air_Op<"ref", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "ref";
  let description = [{
    Syntax:

    ```
    operation ::= `air.ref` ssa-use `:` type
    ```

    The `ref` operation represents a C++ l-value reference with attachment
    to its source location. Since C++ references are immutable, we don't
    need to allocate a separate location specifically for them and
    can use a separate SSA value for that. The operation takes one argument
    and returns it as-is. It has no standard attributes.
  }];

  let arguments = (ins Air_AnyPointer:$value);
  let results = (outs Air_AnyPointer:$res);

  let assemblyFormat = "attr-dict $value `:` type($value)";
}

def Air_NoOp : Air_Op<"noop", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "noop";
  let description = [{
    Syntax:

    ```
    operation ::= `air.noop` ssa-use `:` type
    ```

    The `noop` operation represents a copy of the operand value, but with
    attachment to the corresponding source location. It can be used in
    situations when we don't want to produce a separate location on stack
    or do an extra load, but still want to report on that location.
    The operation takes one argument and returns it as-is. It has no
    standard attributes.
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$res);

  let assemblyFormat = "attr-dict $value `:` type($value)";
}

def Air_GetFieldPtr : Air_Op<"getfieldptr", [NoSideEffect]> {
  let summary = "getfieldptr";
  let description = [{
    Syntax:

    ```
    operation ::= `air.getfieldptr` ssa-use `:` type `-> type
    ```

    The `getfieldptr` operation retrieves a pointer to a field from
    a pointer to a record.
  }];

  let arguments = (ins Air_PointerTo<Air_AnyRecordRef>:$record);
  let results = (outs Air_AnyPointer:$res);

  let assemblyFormat = "attr-dict $record `:` type($record) `->` type($res)";
}

//===----------------------------------------------------------------------===//
//                                   Constant
//===----------------------------------------------------------------------===//

def Air_ConstantOp : Air_Op<"constant", [NoSideEffect, ConstantLike]> {
  let summary = "constant";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.constant` attribute-value `:` type
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

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "mlir::Attribute":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>,
    OpBuilder<(ins "mlir::Attribute":$value, "mlir::Type":$type),
    [{ build($_builder, $_state, type, value); }]>,
  ];

  let extraClassDeclaration = [{
    mlir::Attribute getValue() { return (*this)->getAttr("value"); }
  }];
}

def Air_UndefOp : Air_Op<"undef", [NoSideEffect]> {
  let results = (outs AnyType:$res);

  let summary = "undefined value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.undef` `:` type
    ```

    This operation has no operands or attributes. It creates an undefined
    value of the specified type.

    Example:
    ```mlir
    // Undefined integer
    %1 = "air.undef" : i32
    ```
  }];

  let assemblyFormat = "attr-dict `:` type($res)";
}

def Air_NullOp : Air_Op<"null", [NoSideEffect]> {
  let results = (outs Air_AnyPointer:$res);

  let summary = "null pointer";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `air.null` `:` type
    ```

    This operation has no operands or attributes. It creates a null pointer
    of the specified type.

    Example:
    ```mlir
    %1 = "air.null" : !air<ptr si32>
    ```
  }];

  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
//                         Type definition/declaration
//===----------------------------------------------------------------------===//

def Air_RecordDeclOp : Air_Op<"decl", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name);

  let summary = "type declaration";
  let description = [{
    Syntax:

    ```
    `air.decl`
    ```

    This operation has no operands and simply declares a type. It doesn't have
    standard attributes.

    Example:
    ```mlir
    "air.decl" @"a::A"
    ```
  }];

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    static RecordDeclOp create(mlir::Location Loc, llvm::StringRef Name);
  }];
}

def Air_RecordDefOp : Air_Op<"def", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<Air_RecordType>:$type);

  let summary = "type definition";
  let description = [{
    Syntax:

    ```
    `air.def`
    ```

    This operation has defines a named type. It doesn't have standard attributes.

    Example:
    ```mlir
    "air.def" @"a::A" : "air<rec <b:si32>>"
    ```
  }];

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    static RecordDefOp create(mlir::Location Loc, llvm::StringRef Name, RecordType Record);
  }];
}

#endif /* AIROPS_H */
