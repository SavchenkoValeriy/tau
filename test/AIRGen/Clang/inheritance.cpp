// RUN: tau-cc -dump=air %s -- -std=c++17 > %t.out 2>&1
// RUN: FileCheck %s < %t.out

struct A {
  int a;
};
// CHECK-LABEL: air.def @A : !air.rec<><a : si32>

struct B : public A {
  int b;
};
// CHECK-LABEL: air.def @B : !air.rec<!air<recref @A>><b : si32>

struct C : public B {
  int c;
};
// CHECK-LABEL: air.def @C : !air.rec<!air<recref @B>><c : si32>

struct D {
  int d;
};
// CHECK-LABEL: air.def @D : !air.rec<><d : si32>

struct E : public C, public D {
  int e;
};
// CHECK-LABEL: air.def @E : !air.rec<!air<recref @C>, !air<recref @D>><e : si32>

int fooA(A &a) { return a.a; }

int fooB(B &b) { return b.b; }

int fooC(C &c) { return c.c; }

int fooD(D &d) { return d.d; }

int fooE(E &e) { return e.e; }

int test() {
  int result = 0;
  B b;
  result += fooA(b) + fooB(b);
  C c;
  result += fooA(c) + fooB(c) + fooC(c);
  E e;
  result += fooA(e) + fooB(e) + fooC(e) + fooD(e) + fooE(e);
  return result;
}
// CHECK-LABEL:  @"int test()"
// CHECK-DAG:      %[[#B:]] = air.alloca : !air<ptr !air<recref @B>>
// CHECK:          %[[#BA:]] = air.tobase %[[#B]] : !air<ptr !air<recref @B>> to !air<ptr !air<recref @A>>
// CHECK-DAG:      call @"int fooA(A &)"(%[[#BA]])
// CHECK-DAG:      call @"int fooB(B &)"(%[[#B]])
//
// CHECK-DAG:      %[[#C:]] = air.alloca : !air<ptr !air<recref @C>>
// TODO: C doesn't have A as its subobject, so we should
//       convert C to B and B to A to reflect how they layed out
//       in memory
// CHECK-DAG:      %[[#CA:]] = air.tobase %[[#C]] : !air<ptr !air<recref @C>> to !air<ptr !air<recref @A>>
// CHECK-DAG:      call @"int fooA(A &)"(%[[#CA]])
// CHECK-DAG:      %[[#CB:]] = air.tobase %[[#C]] : !air<ptr !air<recref @C>> to !air<ptr !air<recref @B>>
// CHECK-DAG:      call @"int fooB(B &)"(%[[#CB]])
// CHECK-DAG:      call @"int fooC(C &)"(%[[#C]])
//
// CHECK-DAG:      %[[#E:]] = air.alloca : !air<ptr !air<recref @E>>
// TODO: E doesn't have A as its subobject, so we should
//       convert E to C, C to B, and B to A to reflect how they
//       layed out in memory
// CHECK-DAG:      %[[#EA:]] = air.tobase %[[#E]] : !air<ptr !air<recref @E>> to !air<ptr !air<recref @A>>
// CHECK-DAG:      call @"int fooA(A &)"(%[[#EA]])
// TODO: it should be ((E -> C) -> B)
// CHECK-DAG:      %[[#EB:]] = air.tobase %[[#E]] : !air<ptr !air<recref @E>> to !air<ptr !air<recref @B>>
// CHECK-DAG:      call @"int fooB(B &)"(%[[#EB]])
// CHECK-DAG:      %[[#EC:]] = air.tobase %[[#E]] : !air<ptr !air<recref @E>> to !air<ptr !air<recref @C>>
// CHECK-DAG:      call @"int fooC(C &)"(%[[#EC]])
// CHECK-DAG:      call @"int fooE(E &)"(%[[#E]])
