// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

void test_var() { int b; }
// CHECK:       func.func @"void test_var()"() {
// CHECK-NEXT:    %0 = air.undef : si32
// CHECK-NEXT:    %1 = air.alloca : !air.ptr<si32>
// CHECK-NEXT:    air.store %0 -> %1 : !air.ptr<si32>
// CHECK:         return
// CHECK-NEXT:  }

void test_vars() { int a, b; }
// CHECK:       func.func @"void test_vars()"() {
// CHECK-NEXT:    %0 = air.undef : si32
// CHECK-NEXT:    %1 = air.alloca : !air.ptr<si32>
// CHECK-NEXT:    air.store %0 -> %1 : !air.ptr<si32>
// CHECK-NEXT:    %2 = air.undef : si32
// CHECK-NEXT:    %3 = air.alloca : !air.ptr<si32>
// CHECK-NEXT:    air.store %2 -> %3 : !air.ptr<si32>
// CHECK:         return
// CHECK-NEXT:  }

void test_param(int a) { return; }
// CHECK:       func.func @"void test_param(int a)"(%arg0: si32) {
// CHECK-NEXT:    %0 = air.alloca : !air.ptr<si32>
// CHECK-NEXT:    air.store %arg0 -> %0 : !air.ptr<si32>
// CHECK:         return
// CHECK-NEXT:  }
