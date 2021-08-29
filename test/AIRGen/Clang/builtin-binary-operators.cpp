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

int test_sub(int a, int b) { return a - b; }
// CHECK:       builtin.func @"int test_sub(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK-NEXT:    %0 = subi %arg0, %arg1 : si32
// CHECK-NEXT:    return %0 : si32
// CHECK-NEXT:  }

float test_sub(float a, float b) { return a - b; }
// CHECK:       builtin.func @"float test_sub(float a, float b)"(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = subf %arg0, %arg1 : f32
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }

int test_shl(int a, int b) { return a << b; }
// CHECK:       builtin.func @"int test_shl(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK-NEXT:    %0 = shift_left %arg0, %arg1 : si32
// CHECK-NEXT:    return %0 : si32
// CHECK-NEXT:  }

int test_shr(int a, int b) { return a >> b; }
// CHECK:       builtin.func @"int test_shr(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK-NEXT:    %0 = shift_right_signed %arg0, %arg1 : si32
// CHECK-NEXT:    return %0 : si32
// CHECK-NEXT:  }

unsigned test_shr(unsigned a, unsigned b) { return a >> b; }
// FIXME: it should be 'shift_right_unsigned'
// CHECK:       builtin.func @"unsigned int test_shr(unsigned int a, unsigned int b)"(%arg0: ui32, %arg1: ui32) -> ui32 {
// CHECK-NEXT:    %0 = shift_right_signed %arg0, %arg1 : ui32
// CHECK-NEXT:    return %0 : ui32
// CHECK-NEXT:  }

int test_and(int a, int b) { return a & b; }
// CHECK:       builtin.func @"int test_and(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK-NEXT:    %0 = and %arg0, %arg1 : si32
// CHECK-NEXT:    return %0 : si32
// CHECK-NEXT:  }

int test_xor(int a, int b) { return a ^ b; }
// CHECK:       builtin.func @"int test_xor(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK-NEXT:    %0 = xor %arg0, %arg1 : si32
// CHECK-NEXT:    return %0 : si32
// CHECK-NEXT:  }

int test_or(int a, int b) { return a | b; }
// CHECK:       builtin.func @"int test_or(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK-NEXT:    %0 = or %arg0, %arg1 : si32
// CHECK-NEXT:    return %0 : si32
// CHECK-NEXT:  }

int test_nested(int a, int b, int c) {
  return (a + b * c) - (c ^ a + b);
}
// CHECK:      builtin.func @"int test_nested(int a, int b, int c)"(%arg0: si32, %arg1: si32, %arg2: si32) -> si32 {
// CHECK-NEXT:    %0 = muli %arg1, %arg2 : si32
// CHECK-NEXT:    %1 = addi %arg0, %0 : si32
// CHECK-NEXT:    %2 = addi %arg0, %arg1 : si32
// CHECK-NEXT:    %3 = xor %arg2, %2 : si32
// CHECK-NEXT:    %4 = subi %1, %3 : si32
// CHECK-NEXT:    return %4 : si32
// CHECK-NEXT:  }
