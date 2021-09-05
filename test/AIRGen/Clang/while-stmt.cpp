// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

void test_basic(int a) {
  // CHECK-LABEL:   @"void test_basic(int a)"
  // CHECK-DAG:       br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK-DAG:       %[[#COND:]] = air.slt
  // CHECK-DAG:       air.cond_br %[[#COND]], ^bb[[#BODY:]], ^bb[[#NEXT:]]
  while (a < 42)
    // CHECK-DAG:     ^bb[[#BODY]]:
    // CHECK-DAG:       air.addi
    // CHECK-DAG:       br ^bb[[#HEADER]]
    ++a;
  // CHECK-DAG:     ^bb[[#NEXT]]:
  // CHECK-DAG:       br ^bb[[#EXIT:]]
}

void test_empty(int a) {
  // CHECK-LABEL:   @"void test_empty(int a)"
  // CHECK-DAG:       br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK-DAG:       %[[#COND:]] = air.slt
  // CHECK-DAG:       air.cond_br %[[#COND]], ^bb[[#BODY:]], ^bb[[#NEXT:]]
  while (a < 42) {
    // CHECK-DAG:     ^bb[[#BODY]]:
    // CHECK-DAG:       br ^bb[[#HEADER]]
  }
  // CHECK-DAG:     ^bb[[#NEXT]]:
  // CHECK-DAG:       br ^bb[[#EXIT:]]
}

void test_null(int a) {
  // CHECK-LABEL:   @"void test_null(int a)"
  // CHECK-DAG:       br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK-DAG:       %[[#COND:]] = air.slt
  // CHECK-DAG:       air.cond_br %[[#COND]], ^bb[[#BODY:]], ^bb[[#NEXT:]]
  while (a < 42)
    // CHECK-DAG:     ^bb[[#BODY]]:
    // CHECK-DAG:       br ^bb[[#HEADER]]
    ;
  // CHECK-DAG:     ^bb[[#NEXT]]:
  // CHECK-DAG:       br ^bb[[#EXIT:]]
}

void test_decl(int a) {
  // CHECK-LABEL:   @"void test_decl(int a)"
  // CHECK-DAG:       br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK-DAG:       %[[#COND:]] = air.slt
  // CHECK-DAG:       air.store %[[#COND]] -> %[[#COND_VAR:]]
  // CHECK-DAG:       %[[#COND:]] = air.load %[[#COND_VAR]]
  // CHECK-DAG:       air.cond_br %[[#COND]], ^bb[[#BODY:]], ^bb[[#NEXT:]]
  while (bool cond = a < 42)
    // CHECK-DAG:     ^bb[[#BODY]]:
    // CHECK-DAG:       air.addi
    // CHECK-DAG:       br ^bb[[#HEADER]]
    ++a;
  // CHECK-DAG:     ^bb[[#NEXT]]:
  // CHECK-DAG:       br ^bb[[#EXIT:]]
}