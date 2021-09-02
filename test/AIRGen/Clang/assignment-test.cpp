// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

void test_simple() {
  int a = 0;
  a = 42;
}
// CHECK-LABEL:   builtin.func @"void test_simple()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<si32>
// CHECK:           %[[#CONST:]] = air.constant 42 : si32
// CHECK-NEXT:      air.store %[[#CONST]] -> %[[#A]] : !air.ptr<si32>

void test_addi() {
  int a = 0;
  a += 42;
}
// CHECK-LABEL:   builtin.func @"void test_addi()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<si32>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<si32>
// CHECK-DAG:       %[[#CONST:]] = air.constant 42 : si32
// CHECK-NEXT:      %[[#RES:]] = air.addi %[[#INIT]], %[[#CONST]] : si32
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<si32>

void test_addf() {
  double a = 0.0;
  a += 36.6;
}
// CHECK-LABEL:   builtin.func @"void test_addf()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<f64>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<f64>
// CHECK-DAG:       %[[FCONST:[^:blank]+]] = constant 3.660000e+01 : f64
// CHECK-NEXT:      %[[#RES:]] = addf %[[#INIT]], %[[FCONST]] : f64
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<f64>

void test_subi() {
  int a = 0;
  a -= 42;
}
// CHECK-LABEL:   builtin.func @"void test_subi()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<si32>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<si32>
// CHECK-DAG:       %[[#CONST:]] = air.constant 42 : si32
// CHECK-NEXT:      %[[#RES:]] = air.subi %[[#INIT]], %[[#CONST]] : si32
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<si32>

void test_subf() {
  double a = 0.0;
  a -= 36.6;
}
// CHECK-LABEL:   builtin.func @"void test_subf()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<f64>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<f64>
// CHECK-DAG:       %[[FCONST:[^:blank]+]] = constant 3.660000e+01 : f64
// CHECK-NEXT:      %[[#RES:]] = subf %[[#INIT]], %[[FCONST]] : f64
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<f64>

void test_muli() {
  int a = 0;
  a *= 42;
}
// CHECK-LABEL:   builtin.func @"void test_muli()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<si32>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<si32>
// CHECK-DAG:       %[[#CONST:]] = air.constant 42 : si32
// CHECK-NEXT:      %[[#RES:]] = air.muli %[[#INIT]], %[[#CONST]] : si32
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<si32>

void test_mulf() {
  double a = 0.0;
  a *= 36.6;
}
// CHECK-LABEL:   builtin.func @"void test_mulf()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<f64>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<f64>
// CHECK-DAG:       %[[FCONST:[^:blank]+]] = constant 3.660000e+01 : f64
// CHECK-NEXT:      %[[#RES:]] = mulf %[[#INIT]], %[[FCONST]] : f64
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<f64>

void test_xor() {
  int a = 0;
  a ^= 42;
}
// CHECK-LABEL:   builtin.func @"void test_xor()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<si32>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<si32>
// CHECK-DAG:       %[[#CONST:]] = air.constant 42 : si32
// CHECK-NEXT:      %[[#RES:]] = air.xor %[[#INIT]], %[[#CONST]] : si32
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<si32>

void test_or() {
  int a = 0;
  a |= 42;
}
// CHECK-LABEL:   builtin.func @"void test_or()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<si32>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<si32>
// CHECK-DAG:       %[[#CONST:]] = air.constant 42 : si32
// CHECK-NEXT:      %[[#RES:]] = air.or %[[#INIT]], %[[#CONST]] : si32
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<si32>

void test_and() {
  int a = 0;
  a &= 42;
}
// CHECK-LABEL:   builtin.func @"void test_and()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<si32>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<si32>
// CHECK-DAG:       %[[#CONST:]] = air.constant 42 : si32
// CHECK-NEXT:      %[[#RES:]] = air.and %[[#INIT]], %[[#CONST]] : si32
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<si32>

void test_shl() {
  int a = 0;
  a <<= 5;
}
// CHECK-LABEL:   builtin.func @"void test_shl()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<si32>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<si32>
// CHECK-DAG:       %[[#CONST:]] = air.constant 5 : si32
// CHECK-NEXT:      %[[#RES:]] = air.shl %[[#INIT]], %[[#CONST]] : si32
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<si32>

void test_shr() {
  int a = 512;
  a >>= 5;
}
// CHECK-LABEL:   builtin.func @"void test_shr()"()
// CHECK:           %[[#A:]] = air.alloca : !air.ptr<si32>
// CHECK-DAG:       %[[#INIT:]] = air.load %[[#A]] : !air.ptr<si32>
// CHECK-DAG:       %[[#CONST:]] = air.constant 5 : si32
// CHECK-NEXT:      %[[#RES:]] = air.ashr %[[#INIT]], %[[#CONST]] : si32
// CHECK-NEXT:      air.store %[[#RES]] -> %[[#A]] : !air.ptr<si32>
