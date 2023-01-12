// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_pre_inc(int a) { return ++a; }
// CHECK-LABEL: @"int test_pre_inc(int)"
// CHECK-NEXT:    %[[#A:]] = air.alloca : !air<ptr si32>
// CHECK-DAG:     %[[#ONE:]] = air.constant 1 : si32
// CHECK-DAG:     %[[#VAL:]] = air.load %[[#A]] : !air<ptr si32>
// CHECK-NEXT:    %[[#SUM:]] = air.addi %[[#VAL]], %[[#ONE]] : si32
// CHECK-NEXT:    air.store %[[#SUM]] -> %[[#A]] : !air<ptr si32>
// CHECK-NEXT:    %[[#VAL:]] = air.load %[[#A]] : !air<ptr si32>
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#VAL]] : si32)

int test_post_inc(int a) { return a++; }
// CHECK-LABEL: @"int test_post_inc(int)"
// CHECK-NEXT:    %[[#A:]] = air.alloca : !air<ptr si32>
// CHECK-DAG:     %[[#ONE:]] = air.constant 1 : si32
// CHECK-DAG:     %[[#VAL:]] = air.load %[[#A]] : !air<ptr si32>
// CHECK-NEXT:    %[[#SUM:]] = air.addi %[[#VAL]], %[[#ONE]] : si32
// CHECK-NEXT:    air.store %[[#SUM]] -> %[[#A]] : !air<ptr si32>
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#VAL]] : si32)

float test_pre_inc(float a) { return ++a; }
// CHECK-LABEL: @"float test_pre_inc(float)"
// CHECK-NEXT:    %[[#A:]] = air.alloca : !air<ptr f32>
// CHECK-DAG:     %[[#ONE:]] = air.constant 1.000000e+00 : f32
// CHECK-DAG:     %[[#VAL:]] = air.load %[[#A]] : !air<ptr f32>
// CHECK-NEXT:    %[[#SUM:]] = arith.addf %[[#VAL]], %[[#ONE]] : f32
// CHECK-NEXT:    air.store %[[#SUM]] -> %[[#A]] : !air<ptr f32>
// CHECK-NEXT:    %[[#VAL:]] = air.load %[[#A]] : !air<ptr f32>
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#VAL]] : f32)

float test_post_inc(float a) { return a++; }
// CHECK-LABEL: @"float test_post_inc(float)"
// CHECK-NEXT:    %[[#A:]] = air.alloca : !air<ptr f32>
// CHECK-DAG:     %[[#ONE:]] = air.constant 1.000000e+00 : f32
// CHECK-DAG:     %[[#VAL:]] = air.load %[[#A]] : !air<ptr f32>
// CHECK-NEXT:    %[[#SUM:]] = arith.addf %[[#VAL]], %[[#ONE]] : f32
// CHECK-NEXT:    air.store %[[#SUM]] -> %[[#A]] : !air<ptr f32>
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#VAL]] : f32)
