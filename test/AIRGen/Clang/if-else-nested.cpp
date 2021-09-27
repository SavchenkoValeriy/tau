// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_then_nested(int a) {
  int b;
  // CHECK-LABEL:   @"int test_then_nested(int a)"
  // CHECK-DAG:       %[[#UNDEF:]] = air.undef
  // CHECK-DAG:       air.store %[[#UNDEF]] -> %[[#B:]]
  // CHECK-DAG:       %[[#COND:]] = air.sgt %[[#LHS:]], %[[#RHS:]] : si32
  // CHECK-NEXT:      air.cond_br %[[#COND]], ^bb[[#THEN:]], ^bb[[#ELSE:]]
  if (a > 42) {
    // CHECK-DAG:     ^bb[[#THEN]]:
    // CHECK-DAG:       %[[#COND:]] = air.slt
    // CHECK-DAG:       air.cond_br %[[#COND]], ^bb[[#THENNEST:]], ^bb[[#NEXTNEST:]]
    if (a < 99)
      // CHECK-DAG:     ^bb[[#THENNEST]]:
      // CHECK-DAG:       air.store %[[#A:]] -> %[[#B]]
      // CHECK-DAG:       br ^bb[[#NEXTNEST]]
      b = a;
    // CHECK-DAG:     ^bb[[#NEXTNEST]]:
    // CHECK-DAG:       br ^bb[[#NEXT:]]
  }
  // CHECK-DAG:     ^bb[[#ELSE]]:
  // CHECK-DAG:       %[[#CONST:]] = air.constant 42
  // CHECK-DAG:       air.store %[[#CONST]] -> %[[#B]]
  // CHECK-DAG:       br ^bb[[#NEXT]]
  else
    b = 42;
  // CHECK-DAG:     ^bb[[#NEXT]]:
  // CHECK-DAG:       %[[#RES:]] = air.load %[[#B]]
  // CHECK-DAG:       br ^bb[[#EXIT:]](%[[#RES:]] : si32)
  return b;
}

int test_else_nested(int a) {
  int b;
  // CHECK-LABEL:   @"int test_else_nested(int a)"
  // CHECK-DAG:       %[[#UNDEF:]] = air.undef
  // CHECK-DAG:       air.store %[[#UNDEF]] -> %[[#B:]]
  // CHECK-DAG:       %[[#COND:]] = air.sgt %[[#LHS:]], %[[#RHS:]] : si32
  // CHECK-NEXT:      air.cond_br %[[#COND]], ^bb[[#THEN:]], ^bb[[#ELSE:]]
  if (a > 42) {
    // CHECK-DAG:     ^bb[[#THEN]]:
    // CHECK-NEXT:      %[[#A:]] = air.load
    // CHECK-NEXT:      air.store %[[#A]] -> %[[#B]]
    // CHECK-NEXT:      br ^bb[[#NEXT:]]
    b = a;
  }
  // CHECK-DAG:     ^bb[[#ELSE]]:
  // CHECK-DAG:       %[[#COND:]] = air.slt
  // CHECK-DAG:       air.cond_br %[[#COND]], ^bb[[#THENNEST:]], ^bb[[#ELSENEST:]]
  else if (a < 99)
    // CHECK-DAG:     ^bb[[#THENNEST]]:
    // CHECK-DAG:       %[[#CONST:]] = air.constant 42
    // CHECK-DAG:       air.store %[[#CONST]] -> %[[#B]]
    // CHECK-DAG:       br ^bb[[#NEXTNEST:]]
    b = 42;
  else
    // CHECK-DAG:     ^bb[[#ELSENEST]]:
    // CHECK-DAG:       %[[#CONST:]] = air.constant 0
    // CHECK-DAG:       air.store %[[#CONST]] -> %[[#B]]
    // CHECK-DAG:       br ^bb[[#NEXTNEST:]]
    b = 0;
    // CHECK-DAG:     ^bb[[#NEXTNEST]]:
    // CHECK-DAG:       br ^bb[[#NEXT:]]

  // CHECK-DAG:     ^bb[[#NEXT]]:
  // CHECK-DAG:       %[[#RES:]] = air.load %[[#B]]
  // CHECK-DAG:       br ^bb[[#EXIT:]](%[[#RES:]] : si32)
  return b;
}
