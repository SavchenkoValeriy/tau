// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_add(int a, int b) { return a + b; }
// CHECK:       builtin.func @"int test_add(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.addi %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    return %[[#RES]] : si32
// CHECK-NEXT:  }

long test_add(long a, long b) { return a + b; }
// CHECK:       builtin.func @"long test_add(long a, long b)"(%arg0: si64, %arg1: si64) -> si64 {
// CHECK:         %[[#RES:]] = air.addi %[[#LHS:]], %[[#RHS:]] : si64
// CHECK-NEXT:    return %[[#RES]] : si64
// CHECK-NEXT:  }

float test_add(float a, float b) { return a + b; }
// CHECK:       builtin.func @"float test_add(float a, float b)"(%arg0: f32, %arg1: f32) -> f32 {
// CHECK:         %[[#RES:]] = addf %[[#LHS:]], %[[#RHS:]] : f32
// CHECK-NEXT:    return %[[#RES]] : f32
// CHECK-NEXT:  }

int test_mul(int a, int b) { return a * b; }
// CHECK:       builtin.func @"int test_mul(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.muli %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    return %[[#RES]] : si32
// CHECK-NEXT:  }

unsigned test_mul(unsigned a, unsigned b) { return a * b; }
// CHECK:       builtin.func @"unsigned int test_mul(unsigned int a, unsigned int b)"(%arg0: ui32, %arg1: ui32) -> ui32 {
// CHECK:         %[[#RES:]] = air.muli %[[#LHS:]], %[[#RHS:]] : ui32
// CHECK-NEXT:    return %[[#RES]] : ui32
// CHECK-NEXT:  }

double test_mul(double a, double b) { return a * b; }
// CHECK:       builtin.func @"double test_mul(double a, double b)"(%arg0: f64, %arg1: f64) -> f64 {
// CHECK:         %[[#RES:]] = mulf %[[#LHS:]], %[[#RHS:]] : f64
// CHECK-NEXT:    return %[[#RES]] : f64
// CHECK-NEXT:  }

int test_sub(int a, int b) { return a - b; }
// CHECK:       builtin.func @"int test_sub(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.subi %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    return %[[#RES]] : si32
// CHECK-NEXT:  }

float test_sub(float a, float b) { return a - b; }
// CHECK:       builtin.func @"float test_sub(float a, float b)"(%arg0: f32, %arg1: f32) -> f32 {
// CHECK:         %[[#RES:]] = subf %[[#LHS:]], %[[#RHS:]] : f32
// CHECK-NEXT:    return %[[#RES]] : f32
// CHECK-NEXT:  }

int test_shl(int a, int b) { return a << b; }
// CHECK:       builtin.func @"int test_shl(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.shl %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    return %[[#RES]] : si32
// CHECK-NEXT:  }

int test_shr(int a, int b) { return a >> b; }
// CHECK:       builtin.func @"int test_shr(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.ashr %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    return %[[#RES]] : si32
// CHECK-NEXT:  }

unsigned test_shr(unsigned a, unsigned b) { return a >> b; }
// FIXME: it should be 'shift_right_unsigned'
// CHECK:       builtin.func @"unsigned int test_shr(unsigned int a, unsigned int b)"(%arg0: ui32, %arg1: ui32) -> ui32 {
// CHECK:         %[[#RES:]] = air.ashr %[[#LHS:]], %[[#RHS:]] : ui32
// CHECK-NEXT:    return %[[#RES]] : ui32
// CHECK-NEXT:  }

int test_and(int a, int b) { return a & b; }
// CHECK:       builtin.func @"int test_and(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.and %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    return %[[#RES]] : si32
// CHECK-NEXT:  }

int test_xor(int a, int b) { return a ^ b; }
// CHECK:       builtin.func @"int test_xor(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.xor %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    return %[[#RES]] : si32
// CHECK-NEXT:  }

int test_or(int a, int b) { return a | b; }
// CHECK:       builtin.func @"int test_or(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.or %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    return %[[#RES]] : si32
// CHECK-NEXT:  }

int test_comma(int a, int b) { return a, b; }
// CHECK:       builtin.func @"int test_comma(int a, int b)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         air.store %arg1 : si32 -> %[[#B:]] : !air.ptr<si32>
// CHECK-NEXT:    %[[#RES:]] = air.load %[[#B]] : !air.ptr<si32> -> si32
// CHECK-NEXT:    return %[[#RES]] : si32
// CHECK-NEXT:  }

int test_nested(int a, int b, int c) {
  return (a + b * c) - (c ^ a + b);
}
// CHECK:      builtin.func @"int test_nested(int a, int b, int c)"(%arg0: si32, %arg1: si32, %arg2: si32) -> si32 {
// CHECK:         %[[#BC:]] = air.muli %[[#B:]], %[[#C:]] : si32
// CHECK:         %[[#ABC:]] = air.addi %[[#A:]], %[[#BC]] : si32
// CHECK:         %[[#AB:]] = air.addi %[[#A:]], %[[#B:]] : si32
// CHECK:         %[[#CAB:]] = air.xor %[[#C:]], %[[#AB]] : si32
// CHECK-NEXT:    %[[#RES:]] = air.subi %[[#ABC]], %[[#CAB]] : si32
// CHECK-NEXT:    return %[[#RES]] : si32
// CHECK-NEXT:  }
