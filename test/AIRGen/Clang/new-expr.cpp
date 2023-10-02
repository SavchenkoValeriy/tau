// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

namespace a {
class A {
public:
  int x;
};

class B {
public:
  B(int a, int b) {}
};
} // namespace a

void builtin_type_new() { int *x = new int(42); }
// CHECK-LABEL: "void builtin_type_new()"
// CHECK:         %[[#X_PTR:]] = air.alloca
// CHECK:         %[[#X:]] = air.halloca
// CHECK:         %[[#INIT:]] = air.constant 42
// CHECK:         air.store %[[#INIT]] -> %[[#X]]
// CHECK:         air.store %[[#X]] -> %[[#X_PTR]]

void builtin_type_array_new() { int *y = new int[]{1, 2, 3}; }
// TODO: support array initialization

void class_new() { a::A *a = new a::A; }
// CHECK-LABEL: "void class_new()"
// CHECK:         %[[#A_PTR:]] = air.alloca
// CHECK:         %[[#A:]] = air.halloca
// CHECK:         call @"void a::A::A(a::A *)"(%[[#A]])
// CHECK:         air.store %[[#A]] -> %[[#A_PTR]]

void class_new_args() { a::B *b = new a::B(10, 20); }
// CHECK-LABEL: "void class_new_args()"
// CHECK:         %[[#B_PTR:]] = air.alloca
// CHECK:         %[[#B:]] = air.halloca
// CHECK:         %[[#FIRST:]] = air.constant 10
// CHECK:         %[[#SECOND:]] = air.constant 20
// CHECK:         call @"void a::B::B(a::B *, int, int)"(%[[#B]], %[[#FIRST]], %[[#SECOND]])
// CHECK:         air.store %[[#B]] -> %[[#B_PTR]]
