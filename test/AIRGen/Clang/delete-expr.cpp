// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

struct A {
  int x;
  ~A() {}
};

void test_delete() {
  A* a = new A();
  delete a;
}

// CHECK-LABEL: @"void test_delete()"
// CHECK: %[[A:.*]] = air.alloca
// CHECK: %[[NEW:.*]] = air.halloca : !air<ptr !air<recref @A>>
// CHECK: call @"void A::A(A *)"(%[[NEW]])
// CHECK: air.store %[[NEW]] -> %[[A]]
// CHECK: %[[LOAD:.*]] = air.load %[[A]]
// CHECK: call @"void A::~A(A *)"(%[[LOAD]])
// CHECK: air.hdealloca %[[LOAD]]

void test_delete_array() {
  A* arr = new A[5];
  delete[] arr;
}
// TODO: support arrays
