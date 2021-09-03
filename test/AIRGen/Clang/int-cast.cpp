// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_implicit_ui_to_si(unsigned a) { return a; }
// CHECK-LABEL:   @"int test_implicit_ui_to_si(unsigned int a)"
// CHECK:           %[[#RES:]] = air.bitcast %[[#INP:]] : ui32 to si32
// CHECK-NEXT:      return %[[#RES]] : si32

int test_cstyle_ui_to_si(unsigned a) { return (int)a; }
// CHECK-LABEL:   @"int test_cstyle_ui_to_si(unsigned int a)"
// CHECK:           %[[#RES:]] = air.bitcast %[[#INP:]] : ui32 to si32
// CHECK-NEXT:      return %[[#RES]] : si32

int test_static_ui_to_si(unsigned a) { return static_cast<int>(a); }
// CHECK-LABEL:   @"int test_static_ui_to_si(unsigned int a)"
// CHECK:           %[[#RES:]] = air.bitcast %[[#INP:]] : ui32 to si32
// CHECK-NEXT:      return %[[#RES]] : si32
