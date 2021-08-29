// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_add(int a, int b) { return a + b; }
// CHECK:       builtin.func @"int test_add(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK-NEXT:    %0 = addi %arg0, %arg1 : si32
// CHECK-NEXT:    return %0 : si32
// CHECK-NEXT:  }

long test_add(long a, long b) { return a + b; }
// CHECK:       builtin.func @"long test_add(long a, long b)"(%arg0: si64, %arg1: si64) -> si64 {
// CHECK-NEXT:    %0 = addi %arg0, %arg1 : si64
// CHECK-NEXT:    return %0 : si64
// CHECK-NEXT:  }

float test_add(float a, float b) { return a + b; }
// FIXME remove none types here
// CHECK:       builtin.func @"float test_add(float a, float b)"(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = addf %arg0, %arg1 : f32
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }
