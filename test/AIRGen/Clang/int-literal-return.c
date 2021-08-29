// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

// CHECK:   builtin.func @"int foo()"() -> si32 {
int foo() {
// CHECK-NEXT: %[[CONST:[^:blank:]+]] = constant 42 : i32
// CHECK-NEXT: return %[[CONST]] : i32
  return 42;
}
