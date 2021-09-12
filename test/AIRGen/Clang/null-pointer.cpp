// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

void test_int_zero() { int *a = 0; }
// CHECK-LABEL:   @"void test_int_zero()"
// CHECK-DAG:       %[[#NULL:]] = air.null : !air.ptr<si32>
// CHECK-DAG:       air.store %[[#NULL]] -> %[[#A:]]

void test_nullptr() { int *a = nullptr; }
// CHECK-LABEL:   @"void test_nullptr()"
// CHECK-DAG:       %[[#NULL:]] = air.null : !air.ptr<si32>
// CHECK-DAG:       air.store %[[#NULL]] -> %[[#A:]]
