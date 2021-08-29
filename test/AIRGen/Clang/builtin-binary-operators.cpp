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
// CHECK:       builtin.func @"float test_add(float a, float b)"(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = addf %arg0, %arg1 : f32
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }

int test_mul(int a, int b) { return a * b; }
// CHECK:       builtin.func @"int test_mul(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK-NEXT:    %0 = muli %arg0, %arg1 : si32
// CHECK-NEXT:    return %0 : si32
// CHECK-NEXT:  }

unsigned test_mul(unsigned a, unsigned b) { return a * b; }
// CHECK:       builtin.func @"unsigned int test_mul(unsigned int a, unsigned int b)"(%arg0: ui32, %arg1: ui32) -> ui32 {
// CHECK-NEXT:    %0 = muli %arg0, %arg1 : ui32
// CHECK-NEXT:    return %0 : ui32
// CHECK-NEXT:  }

double test_mul(double a, double b) { return a * b; }
// CHECK:       builtin.func @"double test_mul(double a, double b)"(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %0 = mulf %arg0, %arg1 : f64
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }
