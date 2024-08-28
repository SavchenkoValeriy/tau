// RUN: tau-cc -dump=air %s -- -std=c++17 > %t.out 2>&1
// RUN: FileCheck %s < %t.out

void foo();
void bar();

struct A {
  ~A() { foo(); }
};
// CHECK-LABEL: @"void A::~A(A *)"
// CHECK: call @"void foo()"

struct B {
  A a;
};
// CHECK-LABEL: @"void B::~B(B *)"
// CHECK: %[[#BPTR:]] = air.getfieldptr %{{.*}} -> "a" : !air<ptr !air<recref @B>> -> !air<ptr !air<recref @A>>
// CHECK: call @"void A::~A(A *)"(%[[#BPTR]])

struct C : public B {
};
// CHECK-LABEL: @"void C::~C(C *)"
// CHECK: %[[#CBASE:]] = air.tobase %{{.*}} : !air<ptr !air<recref @C>> to !air<ptr !air<recref @B>>
// CHECK: call @"void B::~B(B *)"(%[[#CBASE]])

struct Combined : public C {
  A x;
  B y;
  C z;
  ~Combined() {
    bar();
  }
};
// CHECK-LABEL: @"void Combined::~Combined(Combined *)"
// CHECK: call @"void bar()"
// CHECK: %[[#ZPTR:]] = air.getfieldptr %{{.*}} -> "z" : !air<ptr !air<recref @Combined>> -> !air<ptr !air<recref @C>>
// CHECK: call @"void C::~C(C *)"(%[[#ZPTR]])
// CHECK: %[[#YPTR:]] = air.getfieldptr %{{.*}} -> "y" : !air<ptr !air<recref @Combined>> -> !air<ptr !air<recref @B>>
// CHECK: call @"void B::~B(B *)"(%[[#YPTR]])
// CHECK: %[[#XPTR:]] = air.getfieldptr %{{.*}} -> "x" : !air<ptr !air<recref @Combined>> -> !air<ptr !air<recref @A>>
// CHECK: call @"void A::~A(A *)"(%[[#XPTR]])
// CHECK: %[[#COMBINEDBASE:]] = air.tobase %{{.*}} : !air<ptr !air<recref @Combined>> to !air<ptr !air<recref @C>>
// CHECK: call @"void C::~C(C *)"(%[[#COMBINEDBASE]])

void test_nontrivial_destructor() {
  A a;
}
// CHECK-LABEL: @"void test_nontrivial_destructor()"
// CHECK: %[[#APTR:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK: call @"void A::~A(A *)"(%[[#APTR]])

void test_implicit_destructor() {
  C c;
}
// CHECK-LABEL: @"void test_implicit_destructor()"
// CHECK: %[[#CPTR:]] = air.alloca : !air<ptr !air<recref @C>>
// CHECK: call @"void C::~C(C *)"(%[[#CPTR]])
