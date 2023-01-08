// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_not(int a) { return ~a; }
// CHECK:       func.func @"int test_not(int a)"(%arg0: si32) -> si32 {
// CHECK:         %[[#REG:]] = air.not %[[#OP:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#REG]] : si32)

int test_neg(int a) { return -a; }
// CHECK:       func.func @"int test_neg(int a)"(%arg0: si32) -> si32 {
// CHECK:         %[[#REG:]] = air.negi %[[#OP:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#REG]] : si32)

double test_neg(double a) { return -a; }
// CHECK:       func.func @"double test_neg(double a)"(%arg0: f64) -> f64 {
// CHECK:         %[[#REG:]] = arith.negf %[[#OP:]] : f64
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#REG]] : f64)

int test_unary_plus(int a) { return +a; }
// CHECK:       func.func @"int test_unary_plus(int a)"(%arg0: si32) -> si32 {
// CHECK:         %[[#REG:]] = air.load %[[#OP:]] : !air<ptr si32>
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#REG]] : si32)
