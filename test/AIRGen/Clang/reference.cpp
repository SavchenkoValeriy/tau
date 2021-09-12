// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

void test_param(int &a) { a = 42; }
// CHECK-LABEL:   @"void test_param(int &a)"
// CHECK-DAG:       %[[#REF:]] = air.ref %arg0
// CHECK-DAG:       %[[#CONST:]] = air.constant 42
// CHECK-DAG:       air.store %[[#CONST]] -> %[[#REF]]

void test_var_decl(int a) { int &b = a; }
// CHECK-LABEL:   @"void test_var_decl(int a)"
// CHECK-DAG:       air.store %arg0 -> %[[#A:]]
// CHECK-DAG:       %[[#REF:]] = air.ref %[[#A]]

void test_addrof(int &a) { int *b = &a; }
// CHECK-LABEL:   @"void test_addrof(int &a)"
// CHECK-DAG:       %[[#REF:]] = air.ref %arg0
// CHECK-DAG:       air.store %[[#REF:]] -> %[[#B:]]

void test_from_ptr(int *a) { int &b = *a; }
// CHECK-LABEL:   @"void test_from_ptr(int *a)"
// CHECK-DAG:       air.store %arg0 -> %[[#A:]]
// CHECK-DAG:       %[[#AVAL:]] = air.load %[[#A]]
// CHECK-DAG:       %[[#REF:]] = air.ref %[[#AVAL]]

void test_as_int(int &a, int b) { int c = a + b; }
// CHECK-LABEL:   @"void test_as_int(int &a, int b)"
// CHECK-DAG:       %[[#REF:]] = air.ref %arg0
// CHECK-DAG:       air.store %arg1 -> %[[#BLOC:]]
// CHECK-DAG:       %[[#A:]] = air.load %[[#REF]]
// CHECK-DAG:       %[[#B:]] = air.load %[[#BLOC]]
// CHECK-DAG:       %[[#C:]] = air.addi %[[#A]], %[[#B]]
