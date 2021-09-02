// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

bool test_slt(int a, int b) { return a < b; }
// CHECK-LABEL: @"bool test_slt(int a, int b)"
// CHECK:         %[[#RES:]] = air.slt %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_ult(unsigned a, unsigned b) { return a < b; }
// CHECK-LABEL: @"bool test_ult(unsigned int a, unsigned int b)"
// CHECK:         %[[#RES:]] = air.ult %[[#A:]], %[[#B:]] : ui32
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_ltf(double a, double b) { return a < b; }
// CHECK-LABEL: @"bool test_ltf(double a, double b)"
// CHECK:         %[[#RES:]] = air.ltf %[[#A:]], %[[#B:]] : f64
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_sle(int a, int b) { return a <= b; }
// CHECK-LABEL: @"bool test_sle(int a, int b)"
// CHECK:         %[[#RES:]] = air.sle %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_ule(unsigned a, unsigned b) { return a <= b; }
// CHECK-LABEL: @"bool test_ule(unsigned int a, unsigned int b)"
// CHECK:         %[[#RES:]] = air.ule %[[#A:]], %[[#B:]] : ui32
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_lef(double a, double b) { return a <= b; }
// CHECK-LABEL: @"bool test_lef(double a, double b)"
// CHECK:         %[[#RES:]] = air.lef %[[#A:]], %[[#B:]] : f64
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_sgt(int a, int b) { return a > b; }
// CHECK-LABEL: @"bool test_sgt(int a, int b)"
// CHECK:         %[[#RES:]] = air.sgt %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_ugt(unsigned a, unsigned b) { return a > b; }
// CHECK-LABEL: @"bool test_ugt(unsigned int a, unsigned int b)"
// CHECK:         %[[#RES:]] = air.ugt %[[#A:]], %[[#B:]] : ui32
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_gtf(double a, double b) { return a > b; }
// CHECK-LABEL: @"bool test_gtf(double a, double b)"
// CHECK:         %[[#RES:]] = air.gtf %[[#A:]], %[[#B:]] : f64
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_sge(int a, int b) { return a >= b; }
// CHECK-LABEL: @"bool test_sge(int a, int b)"
// CHECK:         %[[#RES:]] = air.sge %[[#A:]], %[[#B:]] : si32
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_uge(unsigned a, unsigned b) { return a >= b; }
// CHECK-LABEL: @"bool test_uge(unsigned int a, unsigned int b)"
// CHECK:         %[[#RES:]] = air.uge %[[#A:]], %[[#B:]] : ui32
// CHECK-NEXT:    return %[[#RES]] : ui1

bool test_gef(double a, double b) { return a >= b; }
// CHECK-LABEL: @"bool test_gef(double a, double b)"
// CHECK:         %[[#RES:]] = air.gef %[[#A:]], %[[#B:]] : f64
// CHECK-NEXT:    return %[[#RES]] : ui1
