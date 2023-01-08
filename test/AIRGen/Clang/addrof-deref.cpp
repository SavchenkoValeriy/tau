// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_deref(int *a) { return *a; }
// CHECK-LABEL:  @"int test_deref(int *a)"
// CHECK-NEXT:     %[[#A:]] = air.alloca : !air<ptr !air<ptr si32>>
// CHECK:          %[[#A1:]] = air.load %[[#A]]
// CHECK-NEXT:     %[[#A2:]] = air.load %[[#A1]]
// CHECK-NEXT:     br ^bb[[#EXIT:]](%[[#A2]] : si32)

void test_deref_assign(int *a) { *a = 42; }
// CHECK-LABEL:  @"void test_deref_assign(int *a)"
// CHECK-NEXT:     %[[#A:]] = air.alloca : !air<ptr !air<ptr si32>>
// CHECK-DAG:      %[[#A1:]] = air.load %[[#A]]
// CHECK-DAG:      %[[#CONST:]] = air.constant 42
// CHECK-NEXT:     air.store %[[#CONST]] -> %[[#A1]]

void test_addrof(int *a, int b) { a = &b; }
// CHECK-LABEL:  @"void test_addrof(int *a, int b)"
// CHECK-DAG:      %[[#A:]] = air.alloca : !air<ptr !air<ptr si32>>
// CHECK-DAG:      %[[#B:]] = air.alloca : !air<ptr si32>
// CHECK:          air.store %[[#B]] -> %[[#A]]
