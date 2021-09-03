// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_basic(int a) {
  int b;
  // CHECK-LABEL:   @"int test_basic(int a)"
  // CHECK-DAG:       %[[#UNDEF:]] = air.undef
  // CHECK-NEXT:      air.store %[[#UNDEF]] -> %[[#B:]]
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

int test_early_return(int a) {
  // CHECK-LABEL:   @"int test_early_return(int a)"
  // CHECK-DAG:       %[[#COND:]] = air.sgt %[[#LHS:]], %[[#RHS:]] : si32
  // CHECK-NEXT:      air.cond_br %[[#COND]], ^bb[[#THEN:]], ^bb[[#ELSE:]]
  if (a > 42) {
    // CHECK-DAG:     ^bb[[#THEN]]:
    // CHECK-NEXT:      %[[#A:]] = air.load
    // CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#A:]] : si32)
    return a;
  }
  // CHECK-DAG:     ^bb[[#ELSE]]:
  // CHECK-NEXT:      %[[#CONST:]] = air.constant 42
  // CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#CONST:]] : si32)
  return 42;
}
