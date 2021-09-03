// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

// CHECK:   builtin.func @"int foo()"() -> si32 {
int foo() {
// CHECK-NEXT:   %0 = air.constant 42 : si32
// CHECK-NEXT:   br ^bb1(%0 : si32)
// CHECK-NEXT: ^bb1(%1: si32):  // pred: ^bb0
// CHECK-NEXT:   return %1 : si32

  return 42;
}
