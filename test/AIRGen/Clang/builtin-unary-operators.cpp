// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_not(int a) { return ~a; }
// CHECK:       builtin.func @"int test_not(int a)"(%arg0: si32) -> si32 {
// CHECK-NEXT:    %0 = "air.not"(%arg0) : (si32) -> si32
// CHECK-NEXT:    return %0 : si32

int test_neg(int a) { return -a; }
// CHECK:       builtin.func @"int test_neg(int a)"(%arg0: si32) -> si32 {
// CHECK-NEXT:    %0 = "air.negi"(%arg0) : (si32) -> si32
// CHECK-NEXT:    return %0 : si32

double test_neg(double a) { return -a; }
// CHECK:       builtin.func @"double test_neg(double a)"(%arg0: f64) -> f64 {
// CHECK-NEXT:    %0 = negf %arg0 : f64
// CHECK-NEXT:    return %0 : f64
