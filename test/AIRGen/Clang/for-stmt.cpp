// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck --enable-var-scope %s < %t.out

void bar(int x);

void test_basic() {
  // CHECK-LABEL:   @"void test_basic()"
  // CHECK:           %[[#ZERO:]] = air.constant 0
  // CHECK-DAG:       air.store %[[#ZERO]] -> %[[#I:]]
  // CHECK-DAG:       br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK-DAG:       %[[#IVAL:]] = air.load %[[#I]]
  // CHECK-DAG:       %[[#TEN:]] = air.constant 10
  // CHECK-DAG:       %[[#COND:]] = air.slt %[[#IVAL]], %[[#TEN]]
  // CHECK-DAG:       air.cond_br %[[#COND]], ^bb[[#BODY:]], ^bb[[#NEXT:]]
  for (int i = 0; i < 10; ++i) {
    // CHECK-DAG:     ^bb[[#BODY]]:
    // CHECK-DAG:       call @"void bar(int)"
    // CHECK-NEXT:      br ^bb[[#INC:]]
    bar(i);
  }
  // CHECK-DAG:     ^bb[[#INC]]
  // CHECK-DAG:       air.store %[[#INCREMENTED:]] -> %[[#I]]
  // CHECK:           br ^bb[[#HEADER]]
  // CHECK-DAG:     ^bb[[#NEXT]]:
  // CHECK-DAG:       br ^bb[[#EXIT:]]
}

void test_empty() {
  // CHECK-LABEL:   @"void test_empty()"
  // CHECK:           %[[#ZERO:]] = air.constant 0
  // CHECK-DAG:       air.store %[[#ZERO]] -> %[[#I:]]
  // CHECK-DAG:       br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK-DAG:       %[[#IVAL:]] = air.load %[[#I]]
  // CHECK-DAG:       %[[#TEN:]] = air.constant 10
  // CHECK-DAG:       %[[#COND:]] = air.slt %[[#IVAL]], %[[#TEN]]
  // CHECK-DAG:       air.cond_br %[[#COND]], ^bb[[#BODY:]], ^bb[[#NEXT:]]
  for (int i = 0; i < 10; ++i) {
    // CHECK-DAG:     ^bb[[#BODY]]:
    // CHECK-DAG:       br ^bb[[#INC:]]
  }
  // CHECK-DAG:     ^bb[[#INC:]]
  // CHECK-DAG:       air.store %[[#INCREMENTED:]] -> %[[#I]]
  // CHECK:           br ^bb[[#HEADER]]
  // CHECK-DAG:     ^bb[[#NEXT]]:
  // CHECK-DAG:       br ^bb[[#EXIT:]]
}

void test_null() {
  // CHECK-LABEL:   @"void test_null()"
  // CHECK:           %[[#ZERO:]] = air.constant 0
  // CHECK-DAG:       air.store %[[#ZERO]] -> %[[#I:]]
  // CHECK-DAG:       br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK-DAG:       %[[#IVAL:]] = air.load %[[#I]]
  // CHECK-DAG:       %[[#TEN:]] = air.constant 10
  // CHECK-DAG:       %[[#COND:]] = air.slt %[[#IVAL]], %[[#TEN]]
  // CHECK-DAG:       air.cond_br %[[#COND]], ^bb[[#BODY:]], ^bb[[#NEXT:]]
  for (int i = 0; i < 10; ++i)
    ;
  // CHECK-DAG:     ^bb[[#BODY]]:
  // CHECK-DAG:       br ^bb[[#INC:]]
  // CHECK-DAG:     ^bb[[#INC:]]
  // CHECK-DAG:       air.store %[[#INCREMENTED:]] -> %[[#I]]
  // CHECK:           br ^bb[[#HEADER]]
  // CHECK-DAG:     ^bb[[#NEXT]]:
  // CHECK-DAG:       br ^bb[[#EXIT:]]
}

void test_no_condition() {
  // CHECK-LABEL:   @"void test_no_condition()"
  // CHECK:           %[[#ZERO:]] = air.constant 0
  // CHECK-DAG:       air.store %[[#ZERO]] -> %[[#I:]]
  // CHECK-DAG:       br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK-DAG:       br ^bb[[#BODY:]]
  for (int i = 0;; ++i) {
    // CHECK-DAG:     ^bb[[#BODY]]:
    // CHECK-DAG:       call @"void bar(int)"
    // CHECK-NEXT:      br ^bb[[#INC:]]
    bar(i);
  }
  // CHECK-DAG:     ^bb[[#INC]]
  // CHECK-DAG:       air.store %[[#INCREMENTED:]] -> %[[#I]]
  // CHECK:           br ^bb[[#HEADER]]
}

void test_no_increment() {
  // CHECK-LABEL:   @"void test_no_increment()"
  // CHECK:           %[[#ZERO:]] = air.constant 0
  // CHECK-DAG:       air.store %[[#ZERO]] -> %[[#I:]]
  // CHECK-DAG:       br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK-DAG:       br ^bb[[#BODY:]]
  for (int i = 0;;) {
    // CHECK-DAG:     ^bb[[#BODY]]:
    // CHECK-DAG:       call @"void bar(int)"
    // CHECK-NEXT:      br ^bb[[#HEADER]]
    bar(i);
  }
}

void test_cond_variable() {
  // CHECK-LABEL:   @"void test_cond_variable()"
  // CHECK:           %[[#ZERO:]] = air.constant 0
  // CHECK-DAG:       air.store %[[#ZERO]] -> %[[#I:]]
  // CHECK-DAG:       br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK-DAG:       %[[#IVAL:]] = air.load %[[#I]]
  // CHECK-DAG:       %[[#TEN:]] = air.constant 10
  // CHECK-DAG:       %[[#COND:]] = air.slt %[[#IVAL]], %[[#TEN]]
  // CHECK-DAG:       air.store %[[#COND]] -> %[[#A:]]
  // CHECK-DAG:       %[[#AVAL:]] = air.load %[[#A]]
  // CHECK-DAG:       air.cond_br %[[#AVAL]], ^bb[[#BODY:]], ^bb[[#NEXT:]]
  for (int i = 0; bool a = i < 10; ++i) {
    // CHECK-DAG:     ^bb[[#BODY]]:
    // CHECK-DAG:       call @"void bar(int)"
    // CHECK-NEXT:      br ^bb[[#INC:]]
    bar(i);
  }
  // CHECK-DAG:     ^bb[[#INC]]
  // CHECK-DAG:       air.store %[[#INCREMENTED:]] -> %[[#I]]
  // CHECK:           br ^bb[[#HEADER]]
  // CHECK-DAG:     ^bb[[#NEXT]]:
  // CHECK-DAG:       br ^bb[[#EXIT:]]
}

void test_nested() {
  // CHECK-LABEL:   @"void test_nested()"
  // CHECK:           %[[#ZERO:]] = air.constant 0
  // CHECK:           air.store %[[#ZERO]] -> %[[#I:]]
  // CHECK:           br ^bb[[#HEADER:]]
  // CHECK-DAG:     ^bb[[#HEADER]]:
  // CHECK:           %[[#IVAL:]] = air.load %[[#I]]
  // CHECK:           %[[#TEN:]] = air.constant 10
  // CHECK:           %[[#COND:]] = air.slt %[[#IVAL]], %[[#TEN]]
  // CHECK:           air.cond_br %[[#COND]], ^bb[[#BODY:]], ^bb[[#NEXT:]]
  for (int i = 0; i < 10; ++i) {
    // CHECK-DAG:     ^bb[[#BODY]]:
    // CHECK:           %[[#ZERO:]] = air.constant 0
    // CHECK:           air.store %[[#ZERO]] -> %[[#J:]]
    // CHECK:           br ^bb[[#NHEADER:]]
    // CHECK-DAG:     ^bb[[#INC]]:
    // CHECK-DAG:       air.store %[[#INCREMENTED:]] -> %[[#I]]
    // CHECK:           br ^bb[[#HEADER]]
    // CHECK-DAG:     ^bb[[#NEXT]]:
    // CHECK:           br ^bb[[#EXIT:]]
    // CHECK-DAG:     ^bb[[#NHEADER]]:
    // CHECK:           %[[#JVAL:]] = air.load %[[#J]]
    // CHECK:           %[[#COND:]] = air.slt %[[#JVAL]]
    // CHECK:           air.cond_br %[[#COND]], ^bb[[#NBODY:]], ^bb[[#NNEXT:]]
    for (int j = 0; j < i; ++j) {
      // CHECK-DAG:     ^bb[[#NBODY]]:
      // CHECK:           call @"void bar(int)"
      // CHECK-NEXT:      br ^bb[[#NINC:]]
      bar(i * 10 + j);
    }
    // CHECK-DAG:     ^bb[[#NINC]]:
    // CHECK:           air.store %[[#INCREMENTED:]] -> %[[#J]]
    // CHECK:           br ^bb[[#NHEADER]]
    // CHECK-DAG:     ^bb[[#NNEXT]]:
    // CHECK:           br ^bb[[#INC:]]
  }
}
