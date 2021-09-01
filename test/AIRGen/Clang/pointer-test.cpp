// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int *test_simple(int *a) { return a; }
// CHECK:       builtin.func @"int *test_simple(int *a)"(%arg0: !air.ptr<si32>) -> !air.ptr<si32> {
// CHECK-NEXT:    %0 = air.alloca : !air.ptr<!air.ptr<si32>>
// CHECK-NEXT:    air.store %arg0 : !air.ptr<si32> -> %0 : !air.ptr<!air.ptr<si32>>
// CHECK-NEXT:    %1 = air.load %0 : !air.ptr<!air.ptr<si32>> -> !air.ptr<si32>
// CHECK-NEXT:    return %1 : !air.ptr<si32>
// CHECK-NEXT:  }

float ****test_nested(float ****a) { return a; }
// CHECK:       builtin.func @"float ****test_nested(float ****a)"(%arg0: !air.ptr<!air.ptr<!air.ptr<!air.ptr<f32>>>>) -> !air.ptr<!air.ptr<!air.ptr<!air.ptr<f32>>>> {
// CHECK-NEXT:    %0 = air.alloca : !air.ptr<!air.ptr<!air.ptr<!air.ptr<!air.ptr<f32>>>>>
// CHECK-NEXT:    air.store %arg0 : !air.ptr<!air.ptr<!air.ptr<!air.ptr<f32>>>> -> %0 : !air.ptr<!air.ptr<!air.ptr<!air.ptr<!air.ptr<f32>>>>>
// CHECK-NEXT:    %1 = air.load %0 : !air.ptr<!air.ptr<!air.ptr<!air.ptr<!air.ptr<f32>>>>> -> !air.ptr<!air.ptr<!air.ptr<!air.ptr<f32>>>>
// CHECK-NEXT:    return %1 : !air.ptr<!air.ptr<!air.ptr<!air.ptr<f32>>>>
// CHECK-NEXT:  }
