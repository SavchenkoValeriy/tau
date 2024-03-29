// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int *test_simple(int *a) { return a; }
// CHECK:       func.func @"int * test_simple(int *)"(%arg0: !air<ptr si32>) -> !air<ptr si32> {
// CHECK-NEXT:    %0 = air.alloca : !air<ptr !air<ptr si32>>
// CHECK-NEXT:    air.store %arg0 -> %0 : !air<ptr !air<ptr si32>>
// CHECK-NEXT:    %1 = air.load %0 : !air<ptr !air<ptr si32>>
// CHECK-NEXT:    br ^bb[[#EXIT:]](%1 : !air<ptr si32>)

float ****test_nested(float ****a) { return a; }
// CHECK:       func.func @"float **** test_nested(float ****)"(%arg0: !air<ptr !air<ptr !air<ptr !air<ptr f32>>>>) -> !air<ptr !air<ptr !air<ptr !air<ptr f32>>>> {
// CHECK-NEXT:    %0 = air.alloca : !air<ptr !air<ptr !air<ptr !air<ptr !air<ptr f32>>>>>
// CHECK-NEXT:    air.store %arg0 -> %0 : !air<ptr !air<ptr !air<ptr !air<ptr !air<ptr f32>>>>>
// CHECK-NEXT:    %1 = air.load %0 : !air<ptr !air<ptr !air<ptr !air<ptr !air<ptr f32>>>>>
// CHECK-NEXT:    br ^bb[[#EXIT:]](%1 : !air<ptr !air<ptr !air<ptr !air<ptr f32>>>>)
