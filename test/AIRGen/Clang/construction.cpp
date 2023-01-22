// RUN: tau-cc -dump=air %s -- -std=c++17 > %t.out 2>&1
// RUN: FileCheck %s < %t.out

class Trivial {};
// CHECK-LABEL: air.def @Trivial : !air.rec<>
// CHECK-LABEL: "void Trivial::Trivial(Trivial *)"
// CHECK-LABEL: "void Trivial::Trivial(Trivial *, const Trivial &)"
// TODO: generate move constructors and destructors

void trivial_ctor() { Trivial a; }
// CHECK-LABEL: "void trivial_ctor()"
// CHECK:         %[[#A:]] = air.alloca : !air<ptr !air<recref @Trivial>>
// CHECK:         call @"void Trivial::Trivial(Trivial *)"(%[[#A]])

void trivial_copy() {
  Trivial a;
  Trivial b = a;
}
// CHECK-LABEL: "void trivial_copy()"
// CHECK-DAG:     %[[#A:]] = air.alloca : !air<ptr !air<recref @Trivial>>
// CHECK:         call @"void Trivial::Trivial(Trivial *)"(%[[#A]])
// CHECK-DAG:     %[[#B:]] = air.alloca : !air<ptr !air<recref @Trivial>>
// CHECK:         call @"void Trivial::Trivial(Trivial *, const Trivial &)"(%[[#B]], %[[#A]])

struct Simple {
  int x, y = 42;
};
// CHECK-LABEL: air.def @Simple : !air.rec<x : si32, y : si32>
//
// CHECK-LABEL: "void Simple::Simple(Simple *)"
// TODO:          put undef into field x
// CHECK:         %[[#YPTR:]] = air.getfieldptr %[[#THIS:]] -> "y"
// CHECK:         %[[#DEF:]] = air.constant 42 : si32
// CHECK:         air.store %[[#DEF]] -> %[[#YPTR]]
//
// CHECK-LABEL: "void Simple::Simple(Simple *, const Simple &)"
// CHECK-DAG:       %[[#THIS:]] = air.ref %arg0
// CHECK-DAG:       %[[#OTHER:]] = air.ref %arg1
//
// CHECK-DAG:       %[[#THISX:]] = air.getfieldptr %[[#THIS]] -> "x"
// CHECK-DAG:       %[[#OTHERX:]] = air.getfieldptr %[[#OTHER]] -> "x"
// CHECK-DAG:       %[[#X:]] = air.load %[[#OTHERX]]
// CHECK-DAG:       air.store %[[#X]] -> %[[#THISX]]
//
// CHECK-DAG:       %[[#THISY:]] = air.getfieldptr %[[#THIS]] -> "y"
// CHECK-DAG:       %[[#OTHERY:]] = air.getfieldptr %[[#OTHER]] -> "y"
// CHECK-DAG:       %[[#Y:]] = air.load %[[#OTHERY]]
// CHECK-DAG:       air.store %[[#Y]] -> %[[#THISY]]
//
// TODO: generate move constructors and destructors

void simple_ctor() { Simple a; }
// CHECK-LABEL: "void simple_ctor()"
// CHECK:         %[[#A:]] = air.alloca : !air<ptr !air<recref @Simple>>
// CHECK:         call @"void Simple::Simple(Simple *)"(%[[#A]])

void simple_copy() {
  Simple a;
  Simple b = a;
}
// CHECK-LABEL: "void simple_copy()"
// CHECK-DAG:     %[[#A:]] = air.alloca : !air<ptr !air<recref @Simple>>
// CHECK:         call @"void Simple::Simple(Simple *)"(%[[#A]])
// CHECK-DAG:     %[[#B:]] = air.alloca : !air<ptr !air<recref @Simple>>
// CHECK:         call @"void Simple::Simple(Simple *, const Simple &)"(%[[#B]], %[[#A]])

struct NestedTrivial {
  Trivial x;
};
// CHECK-LABEL: air.def @NestedTrivial : !air.rec<x : !air<recref @Trivial>>
//
// CHECK-LABEL: "void NestedTrivial::NestedTrivial(NestedTrivial *)"
// CHECK:         %[[#XPTR:]] = air.getfieldptr %[[#THIS:]] -> "x"
// CHECK:         call @"void Trivial::Trivial(Trivial *)"(%[[#XPTR]])
//
// CHECK-LABEL: "void NestedTrivial::NestedTrivial(NestedTrivial *, const NestedTrivial &)"
// CHECK-DAG:       %[[#THIS:]] = air.ref %arg0
// CHECK-DAG:       %[[#OTHER:]] = air.ref %arg1
//
// CHECK-DAG:       %[[#THISX:]] = air.getfieldptr %[[#THIS]] -> "x"
// CHECK-DAG:       %[[#OTHERX:]] = air.getfieldptr %[[#OTHER]] -> "x"
// CHECK-DAG:       call @"void Trivial::Trivial(Trivial *, const Trivial &)"(%[[#THISX]], %[[#OTHERX]])
//
// TODO: generate move constructors and destructors

void nested_ctor() { NestedTrivial a; }
// CHECK-LABEL: "void nested_ctor()"
// CHECK:         %[[#A:]] = air.alloca : !air<ptr !air<recref @NestedTrivial>>
// CHECK:         call @"void NestedTrivial::NestedTrivial(NestedTrivial *)"(%[[#A]])

void nested_copy() {
  NestedTrivial a;
  NestedTrivial b = a;
}
// CHECK-LABEL: "void nested_copy()"
// CHECK-DAG:     %[[#A:]] = air.alloca : !air<ptr !air<recref @NestedTrivial>>
// CHECK:         call @"void NestedTrivial::NestedTrivial(NestedTrivial *)"(%[[#A]])
// CHECK-DAG:     %[[#B:]] = air.alloca : !air<ptr !air<recref @NestedTrivial>>
// CHECK:         call @"void NestedTrivial::NestedTrivial(NestedTrivial *, const NestedTrivial &)"(%[[#B]], %[[#A]])
