// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

bool test_slt(int a, int b) { return a < b; }
// CHECK-LABEL: @"bool test_slt(int, int)"
// CHECK:         %[[#RES:]] = air.slt %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_ult(unsigned a, unsigned b) { return a < b; }
// CHECK-LABEL: @"bool test_ult(unsigned int, unsigned int)"
// CHECK:         %[[#RES:]] = air.ult %[[#A:]], %[[#B:]] : ui32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_ltf(double a, double b) { return a < b; }
// CHECK-LABEL: @"bool test_ltf(double, double)"
// CHECK:         %[[#RES:]] = air.ltf %[[#A:]], %[[#B:]] : f64
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_sle(int a, int b) { return a <= b; }
// CHECK-LABEL: @"bool test_sle(int, int)"
// CHECK:         %[[#RES:]] = air.sle %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_ule(unsigned a, unsigned b) { return a <= b; }
// CHECK-LABEL: @"bool test_ule(unsigned int, unsigned int)"
// CHECK:         %[[#RES:]] = air.ule %[[#A:]], %[[#B:]] : ui32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_lef(double a, double b) { return a <= b; }
// CHECK-LABEL: @"bool test_lef(double, double)"
// CHECK:         %[[#RES:]] = air.lef %[[#A:]], %[[#B:]] : f64
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_sgt(int a, int b) { return a > b; }
// CHECK-LABEL: @"bool test_sgt(int, int)"
// CHECK:         %[[#RES:]] = air.sgt %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_ugt(unsigned a, unsigned b) { return a > b; }
// CHECK-LABEL: @"bool test_ugt(unsigned int, unsigned int)"
// CHECK:         %[[#RES:]] = air.ugt %[[#A:]], %[[#B:]] : ui32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_gtf(double a, double b) { return a > b; }
// CHECK-LABEL: @"bool test_gtf(double, double)"
// CHECK:         %[[#RES:]] = air.gtf %[[#A:]], %[[#B:]] : f64
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_sge(int a, int b) { return a >= b; }
// CHECK-LABEL: @"bool test_sge(int, int)"
// CHECK:         %[[#RES:]] = air.sge %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_uge(unsigned a, unsigned b) { return a >= b; }
// CHECK-LABEL: @"bool test_uge(unsigned int, unsigned int)"
// CHECK:         %[[#RES:]] = air.uge %[[#A:]], %[[#B:]] : ui32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_gef(double a, double b) { return a >= b; }
// CHECK-LABEL: @"bool test_gef(double, double)"
// CHECK:         %[[#RES:]] = air.gef %[[#A:]], %[[#B:]] : f64
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_eq(int a, int b) { return a == b; }
// CHECK-LABEL: @"bool test_eq(int, int)"
// CHECK:         %[[#RES:]] = air.eqi %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_eq(float a, float b) { return a == b; }
// CHECK-LABEL: @"bool test_eq(float, float)"
// CHECK:         %[[#RES:]] = air.eqf %[[#A:]], %[[#B:]] : f32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_ne(int a, int b) { return a != b; }
// CHECK-LABEL: @"bool test_ne(int, int)"
// CHECK:         %[[#RES:]] = air.nei %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)

bool test_ne(float a, float b) { return a != b; }
// CHECK-LABEL: @"bool test_ne(float, float)"
// CHECK:         %[[#RES:]] = air.nef %[[#A:]], %[[#B:]] : f32
// CHECK-NEXT:    br ^bb[[#EXIT:]](%[[#RES]] : ui1)
