// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

void test_var() { int b; }
// CHECK-LABEL: func.func @"void test_var()"() {
// CHECK-DAG:     %[[#UNDEF:]] = air.undef : si32
// CHECK-DAG:     %[[#B:]] = air.alloca : !air<ptr si32>
// CHECK:         air.store %[[#UNDEF]] -> %[[#B]] : !air<ptr si32>
// CHECK:         return
// CHECK-NEXT:  }

void test_vars() { int a, b; }
// CHECK-LABEL: func.func @"void test_vars()"() {
// CHECK-DAG:     %[[#AINIT:]] = air.undef : si32
// CHECK-DAG:     %[[#A:]] = air.alloca : !air<ptr si32>
// CHECK:         air.store %[[#AINIT]] -> %[[#A]] : !air<ptr si32>
// CHECK-DAG:     %[[#BINIT:]] = air.undef : si32
// CHECK-DAG:     %[[#B:]] = air.alloca : !air<ptr si32>
// CHECK:         air.store %[[#BINIT]] -> %[[#B]] : !air<ptr si32>
// CHECK:         return
// CHECK-NEXT:  }

void test_param(int a) { return; }
// CHECK-LABEL: func.func @"void test_param(int)"(%arg0: si32) {
// CHECK:         %[[#A:]] = air.alloca : !air<ptr si32>
// CHECK:         air.store %arg0 -> %[[#A]] : !air<ptr si32>
// CHECK:         return
// CHECK-NEXT:  }
