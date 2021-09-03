// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

int test_implicit_ui_to_si(unsigned a) { return a; }
// CHECK-LABEL:   @"int test_implicit_ui_to_si(unsigned int a)"
// CHECK:           %[[#RES:]] = air.bitcast %[[#INP:]] : ui32 to si32
// CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#RES]] : si32)

int test_cstyle_ui_to_si(unsigned a) { return (int)a; }
// CHECK-LABEL:   @"int test_cstyle_ui_to_si(unsigned int a)"
// CHECK:           %[[#RES:]] = air.bitcast %[[#INP:]] : ui32 to si32
// CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#RES]] : si32)

int test_static_ui_to_si(unsigned a) { return static_cast<int>(a); }
// CHECK-LABEL:   @"int test_static_ui_to_si(unsigned int a)"
// CHECK:           %[[#RES:]] = air.bitcast %[[#INP:]] : ui32 to si32
// CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#RES]] : si32)

unsigned long test_implicit_ui32_to_ui64(unsigned a) { return a; }
// CHECK-LABEL:   @"unsigned long test_implicit_ui32_to_ui64(unsigned int a)"
// CHECK:           %[[#RES:]] = air.zext %[[#INP:]] : ui32 to ui64
// CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#RES]] : ui64)

long test_implicit_ui32_to_ui64(int a) { return a; }
// CHECK-LABEL:   @"long test_implicit_ui32_to_ui64(int a)"
// CHECK:           %[[#RES:]] = air.sext %[[#INP:]] : si32 to si64
// CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#RES]] : si64)

unsigned long test_cstyle_ui32_to_ui64(unsigned a) { return (unsigned long)a; }
// CHECK-LABEL:   @"unsigned long test_cstyle_ui32_to_ui64(unsigned int a)"
// CHECK:           %[[#RES:]] = air.zext %[[#INP:]] : ui32 to ui64
// CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#RES]] : ui64)

unsigned long test_static_ui32_to_ui64(unsigned a) {
  return static_cast<unsigned long>(a);
}
// CHECK-LABEL:   @"unsigned long test_static_ui32_to_ui64(unsigned int a)"
// CHECK:           %[[#RES:]] = air.zext %[[#INP:]] : ui32 to ui64
// CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#RES]] : ui64)

int test_implicit_si64_to_si32(long a) { return a; }
// CHECK-LABEL:   @"int test_implicit_si64_to_si32(long a)"
// CHECK:           %[[#RES:]] = air.trunc %[[#INP:]] : si64 to si32
// CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#RES]] : si32)

int test_cstyle_si64_to_si32(long a) { return (int)a; }
// CHECK-LABEL:   @"int test_cstyle_si64_to_si32(long a)"
// CHECK:           %[[#RES:]] = air.trunc %[[#INP:]] : si64 to si32
// CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#RES]] : si32)

int test_static_si64_to_si32(long a) { return static_cast<int>(a); }
// CHECK-LABEL:   @"int test_static_si64_to_si32(long a)"
// CHECK:           %[[#RES:]] = air.trunc %[[#INP:]] : si64 to si32
// CHECK-NEXT:      br ^bb[[#EXIT:]](%[[#RES]] : si32)
