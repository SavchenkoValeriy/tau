// RUN: tau-cc -dump=air %s -- -std=c++17 > %t.out 2>&1
// RUN: FileCheck %s < %t.out

struct A {
  int x;
};
// CHECK-LABEL: air.def @A : !air.rec<><x : si32>

void test_trivial_destructor() {
  A a;
}
// CHECK-LABEL: "void test_trivial_destructor()"
// CHECK-DAG:     %[[#A:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:     call @"void A::A(A *)"(%[[#A]])
// CHECK-DAG:     air.dealloca %[[#A]] : !air<ptr !air<recref @A>>
// CHECK:         cf.br ^bb1

void test_compound_statement() {
  {
    A a1;
  }
  int x = 10;
  {
    A a2;
  }
  int z = 30;
}
// CHECK-LABEL: "void test_compound_statement()"
// CHECK-DAG:     %[[#A1:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:     call @"void A::A(A *)"(%[[#A1]])
// CHECK-DAG:     air.dealloca %[[#A1]] : !air<ptr !air<recref @A>>
// CHECK-DAG:     %[[#X:]] = air.alloca : !air<ptr si32>
// CHECK-DAG:     %[[#CONST1:]] = air.constant 10 : si32
// CHECK-DAG:     air.store %[[#CONST1]] -> %[[#X]] : !air<ptr si32>
// CHECK-DAG:     %[[#A2:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:     call @"void A::A(A *)"(%[[#A2]])
// CHECK-DAG:     air.dealloca %[[#A2]] : !air<ptr !air<recref @A>>
// CHECK-DAG:     %[[#Z:]] = air.alloca : !air<ptr si32>
// CHECK-DAG:     %[[#CONST3:]] = air.constant 30 : si32
// CHECK-DAG:     air.store %[[#CONST3]] -> %[[#Z]] : !air<ptr si32>
// CHECK-DAG:     air.dealloca %[[#X]] : !air<ptr si32>
// CHECK-DAG:     air.dealloca %[[#Z]] : !air<ptr si32>
// CHECK:         cf.br ^bb1

void test_for_loop() {
  for (int i = 0; i < 10; ++i) {
    A a;
  }
  int x = 100;
}
// CHECK-LABEL: "void test_for_loop()"
// CHECK-DAG:     %[[#LOOP:]] = air.alloca : !air<ptr si32>
// CHECK-DAG:     %[[#ZERO:]] = air.constant 0 : si32
// CHECK-DAG:     air.store %[[#ZERO]] -> %[[#LOOP]] : !air<ptr si32>
// CHECK:         cf.br ^bb2
// CHECK:       ^bb2:
// CHECK-DAG:     %[[#I:]] = air.load %[[#LOOP]] : !air<ptr si32>
// CHECK-DAG:     %[[#TEN:]] = air.constant 10 : si32
// CHECK-DAG:     %[[#COND:]] = air.slt %[[#I]], %[[#TEN]] : si32
// CHECK:         air.cond_br %[[#COND]], ^bb3, ^bb5
// CHECK:       ^bb3:
// CHECK-DAG:     %[[#A:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:     call @"void A::A(A *)"(%[[#A]])
// CHECK-DAG:     air.dealloca %[[#A]] : !air<ptr !air<recref @A>>
// CHECK:         cf.br ^bb4
// CHECK:       ^bb4:
// CHECK-DAG:     %[[#I:]] = air.load %[[#LOOP]] : !air<ptr si32>
// CHECK-DAG:     %[[#ONE:]] = air.constant 1 : si32
// CHECK-DAG:     %[[#INC:]] = air.addi %[[#I]], %[[#ONE]] : si32
// CHECK-DAG:     air.store %[[#INC]] -> %[[#LOOP]] : !air<ptr si32>
// CHECK:         cf.br ^bb2
// CHECK:       ^bb5:
// CHECK-DAG:     air.dealloca %[[#LOOP]] : !air<ptr si32>
// CHECK-DAG:     %[[#X:]] = air.alloca : !air<ptr si32>
// CHECK-DAG:     %[[#HUNDRED:]] = air.constant 100 : si32
// CHECK-DAG:     air.store %[[#HUNDRED]] -> %[[#X]] : !air<ptr si32>
// CHECK-DAG:     air.dealloca %[[#X]] : !air<ptr si32>
// CHECK:         cf.br ^bb1

void test_if_else() {
  if (true) {
    A a1;
  } else {
    A a2;
  }
  int x = 1;
}
// CHECK-LABEL: "void test_if_else()"
// CHECK-DAG:     %[[#COND:]] = air.constant 1 : ui1
// CHECK:         air.cond_br %[[#COND]], ^bb2, ^bb3
// CHECK:       ^bb2:
// CHECK-DAG:     %[[#A1:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:     call @"void A::A(A *)"(%[[#A1]])
// CHECK-DAG:     air.dealloca %[[#A1]] : !air<ptr !air<recref @A>>
// CHECK:         cf.br ^bb4
// CHECK:       ^bb3:
// CHECK-DAG:     %[[#A2:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:     call @"void A::A(A *)"(%[[#A2]])
// CHECK-DAG:     air.dealloca %[[#A2]] : !air<ptr !air<recref @A>>
// CHECK:         cf.br ^bb4
// CHECK:       ^bb4:
// CHECK-DAG:     %[[#X:]] = air.alloca : !air<ptr si32>
// CHECK-DAG:     %[[#ONE:]] = air.constant 1 : si32
// CHECK-DAG:     air.store %[[#ONE]] -> %[[#X]] : !air<ptr si32>
// CHECK-DAG:     air.dealloca %[[#X]] : !air<ptr si32>
// CHECK:         cf.br ^bb1

void test_while_loop() {
  while (false) {
    A a;
  }
  int x = 5;
}
// CHECK-LABEL: "void test_while_loop()"
// CHECK:         cf.br ^bb2
// CHECK:       ^bb2:
// CHECK-DAG:     %[[#COND:]] = air.constant 0 : ui1
// CHECK:         air.cond_br %[[#COND]], ^bb3, ^bb4
// CHECK:       ^bb3:
// CHECK-DAG:     %[[#A:]] = air.alloca : !air<ptr !air<recref @A>>
// CHECK-DAG:     call @"void A::A(A *)"(%[[#A]])
// CHECK-DAG:     air.dealloca %[[#A]] : !air<ptr !air<recref @A>>
// CHECK:         cf.br ^bb2
// CHECK:       ^bb4:
// CHECK-DAG:     %[[#X:]] = air.alloca : !air<ptr si32>
// CHECK-DAG:     %[[#FIVE:]] = air.constant 5 : si32
// CHECK-DAG:     air.store %[[#FIVE]] -> %[[#X]] : !air<ptr si32>
// CHECK-DAG:     air.dealloca %[[#X]] : !air<ptr si32>
// CHECK:         cf.br ^bb1

void test_parameter_dealloca(A a) {
  int x = a.x;
}
// CHECK-LABEL: @"void test_parameter_dealloca(A)"
// CHECK:       %[[#X:]] = air.alloca : !air<ptr si32>
// CHECK:       air.dealloca %[[#X]] : !air<ptr si32>

void test_if_condition_var_dealloca(int x) {
  if (int y = x * 2; y > 10) {
    A a;
  }
  int z = 0;  // Pivot statement
}
// CHECK-LABEL: @"void test_if_condition_var_dealloca(int)"
// CHECK:       %[[#MUL:]] = air.muli
// CHECK:       air.store %[[#MUL]] -> %[[#Y:]]
// CHECK:       air.cond_br %{{.*}}, ^bb[[#THEN:]], ^bb[[#ELSE:]]
// CHECK:       ^bb[[#NEXT:]]
// CHECK:       air.dealloca %[[#Y]] : !air<ptr si32>
// CHECK-NEXT:  %[[#Z:]] = air.alloca : !air<ptr si32>
// CHECK-NEXT:  %[[#ZERO:]] = air.constant 0 : si32
// CHECK-NEXT:  air.store %[[#ZERO]] -> %[[#Z]] : !air<ptr si32>

void test_for_loop_var_dealloca() {
  for (int i = 0; i < 10; ++i) {
    A a;
  }
  int z = 0;  // Pivot statement
}
// CHECK-LABEL: @"void test_for_loop_var_dealloca()"
// CHECK:       %[[#I:]] = air.alloca : !air<ptr si32>
// CHECK:       %[[#ZERO:]] = air.constant 0 : si32
// CHECK:       air.store %[[#ZERO]] -> %[[#I]] : !air<ptr si32>
// CHECK:       cf.br ^bb[[#LOOP_HEADER:]]
// CHECK:       ^bb[[#LOOP_EXIT:]]
// CHECK:       air.dealloca %[[#I]] : !air<ptr si32>
// CHECK-NEXT:  %[[#Z:]] = air.alloca : !air<ptr si32>
// CHECK-NEXT:  %[[#ZERO:]] = air.constant 0 : si32
// CHECK-NEXT:  air.store %[[#ZERO]] -> %[[#Z]] : !air<ptr si32>
