// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_not(int a) { return ~a; }
// CHECK:       builtin.func @"int test_not(int a)"(%arg0: si32) -> si32 {
// CHECK-NEXT:    %0 = "air.not"(%arg0) : (si32) -> si32
// CHECK-NEXT:    return %0 : si32
