// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

void bar() {}
int foo(int, int, int) { return 42; }

void test_simple() { bar(); }
// CHECK-LABEL:    @"void test_simple()"
// CHECK-DAG:        call @"void bar()"

int test_multiple_args(int a) { return foo(10, a, a + 10); }
// CHECK-LABEL:    @"int test_multiple_args(int a)"
// CHECK-DAG:        %[[#FIRST:]] = air.constant 10
// CHECK-DAG:        %[[#SECOND:]] = air.load
// CHECK-DAG:        %[[#THIRD:]] = air.addi
// CHECK-DAG:        %[[#RES:]] = call @"int foo(int, int, int)"(%[[#FIRST]], %[[#SECOND]], %[[#THIRD]])
// CHECK-DAG:        cf.br ^bb[[#EXIT:]](%[[#RES]] : si32)

int test_recursion() { return test_recursion(); }
// CHECK-LABEL:    @"int test_recursion()"
// CHECK-DAG:        %[[#RES:]] = call @"int test_recursion()"
// CHECK-NEXT:       cf.br ^bb[[#EXIT:]](%[[#RES]] : si32)
