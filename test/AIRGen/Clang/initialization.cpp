// RUN: tau-cc -dump=air %s -- -std=c++17 > %t.out 2>&1
// RUN: FileCheck %s < %t.out

struct A {
  int x;
  float y;
  int z = 42;
};
// CHECK-LABEL:  air.def @A : !air.rec<x : si32, y : f32, z : si32>

int init_list_A(int x, float y) {
  A a{x, y, 10};
  return a.x;
}
// CHECK-LABEL:  @"int init_list_A(int, float)"
// CHECK-DAG:      air.store %arg0 -> %[[#XPTR:]] : !air<ptr si32>
// CHECK-DAG:      air.store %arg1 -> %[[#YPTR:]] : !air<ptr f32>
// CHECK:          %[[#A:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:      %[[#AX:]] = air.getfieldptr %[[#A]] -> "x"
// CHECK:          %[[#X:]] = air.load %[[#XPTR]]
// CHECK:          air.store %[[#X]] -> %[[#AX]]
// CHECK-DAG:      %[[#AY:]] = air.getfieldptr %[[#A]] -> "y"
// CHECK:          %[[#Y:]] = air.load %[[#YPTR]]
// CHECK:          air.store %[[#Y]] -> %[[#AY]]
// CHECK-DAG:      %[[#AZ:]] = air.getfieldptr %[[#A]] -> "z"
// CHECK:          %[[#Z:]] = air.constant 10
// CHECK:          air.store %[[#Z]] -> %[[#AZ]]

int init_list_A_2(int x, float y) {
  A a = {x, y};
  return a.x;
}
// CHECK-LABEL:  @"int init_list_A_2(int, float)"
// CHECK-DAG:      air.store %arg0 -> %[[#XPTR:]] : !air<ptr si32>
// CHECK-DAG:      air.store %arg1 -> %[[#YPTR:]] : !air<ptr f32>
// CHECK:          %[[#A:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:      %[[#AX:]] = air.getfieldptr %[[#A]] -> "x"
// CHECK:          %[[#X:]] = air.load %[[#XPTR]]
// CHECK:          air.store %[[#X]] -> %[[#AX]]
// CHECK-DAG:      %[[#AY:]] = air.getfieldptr %[[#A]] -> "y"
// CHECK:          %[[#Y:]] = air.load %[[#YPTR]]
// CHECK:          air.store %[[#Y]] -> %[[#AY]]
// CHECK-DAG:      %[[#AZ:]] = air.getfieldptr %[[#A]] -> "z"
// CHECK:          %[[#Z:]] = air.constant 42
// CHECK:          air.store %[[#Z]] -> %[[#AZ]]

int init_list_A_3(int x, float y) {
  A a{x};
  return a.x;
}
// CHECK-LABEL:  @"int init_list_A_3(int, float)"
// CHECK-DAG:      air.store %arg0 -> %[[#XPTR:]] : !air<ptr si32>
// CHECK-DAG:      air.store %arg1 -> %[[#YPTR:]] : !air<ptr f32>
// CHECK:          %[[#A:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:      %[[#AX:]] = air.getfieldptr %[[#A]] -> "x"
// CHECK:          %[[#X:]] = air.load %[[#XPTR]]
// CHECK:          air.store %[[#X]] -> %[[#AX]]
// CHECK-DAG:      %[[#AY:]] = air.getfieldptr %[[#A]] -> "y"
// CHECK:          %[[#Y:]] = air.undef
// CHECK:          air.store %[[#Y]] -> %[[#AY]]
// CHECK-DAG:      %[[#AZ:]] = air.getfieldptr %[[#A]] -> "z"
// CHECK:          %[[#Z:]] = air.constant 42
// CHECK:          air.store %[[#Z]] -> %[[#AZ]]

int init_list_A_4(int x, float y) {
  A a{};
  return a.x;
}
// CHECK-LABEL:  @"int init_list_A_4(int, float)"
// CHECK-DAG:      air.store %arg0 -> %[[#XPTR:]] : !air<ptr si32>
// CHECK-DAG:      air.store %arg1 -> %[[#YPTR:]] : !air<ptr f32>
// CHECK:          %[[#A:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:      %[[#AX:]] = air.getfieldptr %[[#A]] -> "x"
// CHECK:          %[[#X:]] = air.undef
// CHECK:          air.store %[[#X]] -> %[[#AX]]
// CHECK-DAG:      %[[#AY:]] = air.getfieldptr %[[#A]] -> "y"
// CHECK:          %[[#Y:]] = air.undef
// CHECK:          air.store %[[#Y]] -> %[[#AY]]
// CHECK-DAG:      %[[#AZ:]] = air.getfieldptr %[[#A]] -> "z"
// CHECK:          %[[#Z:]] = air.constant 42
// CHECK:          air.store %[[#Z]] -> %[[#AZ]]

int no_init_A(int x, float y) {
  A a;
  return a.x;
}
// CHECK-LABEL:  @"int no_init_A(int, float)"
// TODO: generate the same code as for @"int init_list_A_4(int, float)"

int designated_init(int x, float y) {
  A a{.y = y, .x = x};
  return a.x;
}
// CHECK-LABEL:  @"int designated_init(int, float)"
// CHECK-DAG:      air.store %arg0 -> %[[#XPTR:]] : !air<ptr si32>
// CHECK-DAG:      air.store %arg1 -> %[[#YPTR:]] : !air<ptr f32>
// CHECK:          %[[#A:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:      %[[#AX:]] = air.getfieldptr %[[#A]] -> "x"
// CHECK:          %[[#X:]] = air.load %[[#XPTR]]
// CHECK:          air.store %[[#X]] -> %[[#AX]]
// CHECK-DAG:      %[[#AY:]] = air.getfieldptr %[[#A]] -> "y"
// CHECK:          %[[#Y:]] = air.load %[[#YPTR]]
// CHECK:          air.store %[[#Y]] -> %[[#AY]]
// CHECK-DAG:      %[[#AZ:]] = air.getfieldptr %[[#A]] -> "z"
// CHECK:          %[[#Z:]] = air.constant 42
// CHECK:          air.store %[[#Z]] -> %[[#AZ]]
