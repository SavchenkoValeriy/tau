// RUN: tau-cc -dump=air %s -- -std=c++17 > %t.out 2>&1
// RUN: FileCheck %s < %t.out

namespace a::b::c {
class A {
// CHECK-LABEL: air.def @"a::b::c::A" : !air.rec<><x : si32>
public:
  int x;
  int foo(int y) { return x + y; }
  // CHECK-LABEL:  @"int a::b::c::A::foo(a::b::c::A *, int)"
  // CHECK:          %[[#THIS:]] = air.ref %arg0
  // CHECK-DAG:      %[[#XPTR:]] = air.getfieldptr %[[#THIS]] -> "x"
  // CHECK-DAG:      %[[#X:]] = air.load %[[#XPTR]]
  // CHECK-DAG:      %[[#ADD:]] = air.addi %[[#X]], %[[#Y:]]
  A *getThis() { return this; }
  // CHECK-LABEL:  @"a::b::c::A * a::b::c::A::getThis(a::b::c::A *)"
  // CHECK:          %[[#THIS:]] = air.ref %arg0
  // CHECK-DAG:      br ^bb[[#EXIT:]](%[[#THIS]] : !air<ptr !air<recref @"a::b::c::A">>)
};
} // end namespace a::b::c

using namespace a::b;

int bar(int x, int y) {
  c::A a{x};
  return a.foo(y);
}
// CHECK-LABEL:    func.func @"int bar(int, int)"
// CHECK:            %[[#A:]] = air.alloca : !air<ptr !air<recref @"a::b::c::A">>
// CHECK-DAG:        %[[#RES:]] = call @"int a::b::c::A::foo(a::b::c::A *, int)"(%[[#A]], %[[#Y:]])
