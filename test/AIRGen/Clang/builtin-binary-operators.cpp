// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_add(int a, int b) { return a + b; }
// CHECK:       func.func @"int test_add(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.addi %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

long test_add(long a, long b) { return a + b; }
// CHECK:       func.func @"long test_add(long, long)"(%arg0: si64, %arg1: si64) -> si64 {
// CHECK:         %[[#RES:]] = air.addi %[[#LHS:]], %[[#RHS:]] : si64
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si64)

float test_add(float a, float b) { return a + b; }
// CHECK:       func.func @"float test_add(float, float)"(%arg0: f32, %arg1: f32) -> f32 {
// CHECK:         %[[#RES:]] = arith.addf %[[#LHS:]], %[[#RHS:]] : f32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : f32)

int test_mul(int a, int b) { return a * b; }
// CHECK:       func.func @"int test_mul(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.muli %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

unsigned test_mul(unsigned a, unsigned b) { return a * b; }
// CHECK:       func.func @"unsigned int test_mul(unsigned int, unsigned int)"(%arg0: ui32, %arg1: ui32) -> ui32 {
// CHECK:         %[[#RES:]] = air.muli %[[#LHS:]], %[[#RHS:]] : ui32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui32)

double test_mul(double a, double b) { return a * b; }
// CHECK:       func.func @"double test_mul(double, double)"(%arg0: f64, %arg1: f64) -> f64 {
// CHECK:         %[[#RES:]] = arith.mulf %[[#LHS:]], %[[#RHS:]] : f64
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : f64)

int test_div(int a, int b) { return a / b; }
// CHECK:       func.func @"int test_div(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.sdiv %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

unsigned test_div(unsigned a, unsigned b) { return a / b; }
// CHECK:       func.func @"unsigned int test_div(unsigned int, unsigned int)"(%arg0: ui32, %arg1: ui32) -> ui32 {
// CHECK:         %[[#RES:]] = air.udiv %[[#LHS:]], %[[#RHS:]] : ui32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui32)

double test_div(double a, double b) { return a / b; }
// CHECK:       func.func @"double test_div(double, double)"(%arg0: f64, %arg1: f64) -> f64 {
// CHECK:         %[[#RES:]] = arith.divf %[[#LHS:]], %[[#RHS:]] : f64
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : f64)

int test_rem(int a, int b) { return a % b; }
// CHECK:       func.func @"int test_rem(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.srem %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

unsigned test_rem(unsigned a, unsigned b) { return a % b; }
// CHECK:       func.func @"unsigned int test_rem(unsigned int, unsigned int)"(%arg0: ui32, %arg1: ui32) -> ui32 {
// CHECK:         %[[#RES:]] = air.urem %[[#LHS:]], %[[#RHS:]] : ui32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui32)

int test_sub(int a, int b) { return a - b; }
// CHECK:       func.func @"int test_sub(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.subi %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

float test_sub(float a, float b) { return a - b; }
// CHECK:       func.func @"float test_sub(float, float)"(%arg0: f32, %arg1: f32) -> f32 {
// CHECK:         %[[#RES:]] = arith.subf %[[#LHS:]], %[[#RHS:]] : f32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : f32)

int test_shl(int a, int b) { return a << b; }
// CHECK:       func.func @"int test_shl(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.shl %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

int test_shr(int a, int b) { return a >> b; }
// CHECK:       func.func @"int test_shr(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.ashr %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

unsigned test_shr(unsigned a, unsigned b) { return a >> b; }
// CHECK:       func.func @"unsigned int test_shr(unsigned int, unsigned int)"(%arg0: ui32, %arg1: ui32) -> ui32 {
// CHECK:         %[[#RES:]] = air.lshr %[[#LHS:]], %[[#RHS:]] : ui32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui32)

int test_and(int a, int b) { return a & b; }
// CHECK:       func.func @"int test_and(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.and %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

int test_xor(int a, int b) { return a ^ b; }
// CHECK:       func.func @"int test_xor(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.xor %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

int test_or(int a, int b) { return a | b; }
// CHECK:       func.func @"int test_or(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         %[[#RES:]] = air.or %[[#LHS:]], %[[#RHS:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

int test_comma(int a, int b) { return a, b; }
// CHECK:       func.func @"int test_comma(int, int)"(%arg0: si32, %arg1: si32) -> si32 {
// CHECK:         air.store %arg1 -> %[[#B:]] : !air<ptr si32>
// CHECK-NEXT:    %[[#RES:]] = air.load %[[#B]] : !air<ptr si32>
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)

int test_nested(int a, int b, int c) {
  return (a + b * c) - (c ^ a + b);
}
// CHECK:      func.func @"int test_nested(int, int, int)"(%arg0: si32, %arg1: si32, %arg2: si32) -> si32 {
// CHECK:         %[[#BC:]] = air.muli %[[#B:]], %[[#C:]] : si32
// CHECK:         %[[#ABC:]] = air.addi %[[#A:]], %[[#BC]] : si32
// CHECK:         %[[#AB:]] = air.addi %[[#A:]], %[[#B:]] : si32
// CHECK:         %[[#CAB:]] = air.xor %[[#C:]], %[[#AB]] : si32
// CHECK-NEXT:    %[[#RES:]] = air.subi %[[#ABC]], %[[#CAB]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : si32)
