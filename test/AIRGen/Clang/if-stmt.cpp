// RUN: tau-cc -dump=air %s > %t.out 2>&1 -- -std=c++17
// RUN: FileCheck %s < %t.out

int test_basic(int a) {
  int b;
  // CHECK-LABEL:   @"int test_basic(int a)"
  // CHECK-DAG:       %[[#UNDEF:]] = air.undef
  // CHECK-DAG:       air.store %[[#UNDEF]] -> %[[#B:]]
  // CHECK-DAG:       %[[#COND:]] = air.sgt %[[#LHS:]], %[[#RHS:]] : si32
  // CHECK-NEXT:      air.cond_br %[[#COND]], ^bb[[#THEN:]], ^bb[[#ELSE:]]
  if (a > 42) {
    // CHECK-DAG:     ^bb[[#THEN]]:
    // CHECK-DAG:       %[[#A:]] = air.load
    // CHECK-DAG:       air.store %[[#A]] -> %[[#B]]
    // CHECK-DAG:       br ^bb[[#NEXT:]]
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
    // CHECK-DAG:       %[[#A:]] = air.load
    // CHECK-DAG:       br ^bb[[#EXIT:]](%[[#A:]] : si32)
    return a;
  }
  // CHECK-DAG:     ^bb[[#ELSE]]:
  // CHECK-NEXT:      %[[#CONST:]] = air.constant 42
  // CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#CONST:]] : si32)
  return 42;
}

bool test_var_decl(int a) {
  // CHECK-LABEL:   @"bool test_var_decl(int a)"
  // CHECK-DAG:       %[[#EQ:]] = air.eqi
  // CHECK-DAG:       air.store %[[#EQ]] -> %[[#B:]]
  // CHECK-DAG:       %[[#COND:]] = air.load %[[#B]]
  // CHECK-NEXT:      air.cond_br %[[#COND]], ^bb[[#THEN:]], ^bb[[#ELSE:]]
  if (bool b = a == 1) {
    // CHECK-DAG:     ^bb[[#THEN]]:
    // CHECK-NEXT:      %[[#VAL:]] = air.load %[[#B]]
    // CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#VAL]] : ui1)
    return b;
  } else {
    // CHECK-DAG:     ^bb[[#ELSE]]:
    // CHECK-NEXT:      %[[#VAL:]] = air.load %[[#B]]
    // CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#VAL]] : ui1)
    return b;
  }
}

int test_var_decl2(int a) {
  // CHECK-LABEL:   @"int test_var_decl2(int a)"
  // CHECK-DAG:       %[[#ADD:]] = air.addi
  // CHECK-DAG:       air.store %[[#ADD]] -> %[[#B:]]
  // CHECK-DAG:       %[[#COND:]] = air.eqi
  // CHECK-NEXT:      air.cond_br %[[#COND]], ^bb[[#THEN:]], ^bb[[#ELSE:]]
  if (int b = a + 1; b == 42) {
    // CHECK-DAG:     ^bb[[#THEN]]:
    // CHECK-NEXT:      %[[#VAL:]] = air.load %[[#B]]
    // CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#VAL]] : si32)
    return b;
  } else {
    // CHECK-DAG:     ^bb[[#ELSE]]:
    // CHECK-NEXT:      %[[#VAL:]] = air.load %[[#B]]
    // CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#VAL]] : si32)
    return b;
  }
}
