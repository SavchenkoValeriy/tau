// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

void test_var() { int b; }
// CHECK:       builtin.func @"void test_var()"() {
// CHECK-NEXT:    %0 = air.alloca : !air.ptr<si32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

void test_param(int a) { return; }
// CHECK:       builtin.func @"void test_param(int a)"(%arg0: si32) {
// CHECK-NEXT:    %0 = air.alloca : !air.ptr<si32>
// CHECK-NEXT:    air.store %arg0 : si32 -> %0 : !air.ptr<si32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
