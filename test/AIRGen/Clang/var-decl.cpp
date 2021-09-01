// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

void test_var() { int b; }
// CHECK:       builtin.func @"void test_var()"() -> none {
// CHECK-NEXT:    %0 = air.alloca : !air.ptr<si32>
// CHECK-NEXT:  }
