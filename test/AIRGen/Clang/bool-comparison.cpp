// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

bool test_slt(int a, int b) { return a < b; }
// CHECK-LABEL: @"bool test_slt(int a, int b)"
// CHECK:         %[[#RES:]] = air.slt %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    return %[[#RES]] : ui1
