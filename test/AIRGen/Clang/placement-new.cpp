// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

typedef __SIZE_TYPE__ size_t;

extern "C" {
extern void *malloc (__SIZE_TYPE__ __size) throw () __attribute__ ((__malloc__)) ;
}

// Declare the reserved placement operators.
void *operator new(size_t, void*) throw();

// Declare some other placement operators.
void *operator new(size_t, void* mem, bool) throw() {
  return mem;
}

void builtin_type_placement_new() {
  void *buf = malloc(sizeof(int));
  int *x = new (buf) int(42);
}

void builtin_type_placement_new_2() {
  // CHECK-LABEL:  @"void builtin_type_placement_new_2()"
  // CHECK:          %[[#Y:]] = air.alloca
  int y = 10;
  // CHECK:          %[[#X:]] = air.alloca
  //
  // TODO: get rid of casting to void pointer and back
  // CHECK:          %[[#PTR1:]] = air.bitcast %[[#Y]]
  // CHECK:          %[[#PTR2:]] = air.bitcast %[[#PTR1]]
  //
  // CHECK:          %[[#VAL:]] = air.constant 42
  // CHECK:          air.store %[[#VAL]] -> %[[#PTR2]]
  // CHECK:          air.store %[[#PTR2]] -> %[[#X]]
  int *x = new (&y) int(42);
}

void custom_new(void *buf) {
  // CHECK-LABEL:  @"void custom_new(void *)"
  // CHECK:          air.store %arg0 -> %[[#BUFPTR:]]
  // CHECK:          %[[#X:]] = air.alloca : !air<ptr !air<ptr si32>>
  // CHECK:          %[[#BUF:]] = air.load %[[#BUFPTR]]
  // CHECK:          %[[#TRUE:]] = air.constant 1 : ui1
  // CHECK:          %[[#SIZE:]] = air.sizeof si32
  // CHECK:          %[[#RESVOID:]] = call @"void * operator new(unsigned long, void *, bool)"(%[[#SIZE]], %[[#BUF]], %[[#TRUE]])
  // CHECK:          %[[#RES:]] = air.bitcast %[[#RESVOID]] : !air<ptr !air.void> to !air<ptr si32>
  // CHECK:          %[[#INIT:]] = air.constant 42
  // CHECK:          air.store %[[#INIT]] -> %[[#RES]]
  // CHECK:          air.store %[[#RES]] -> %[[#X]]
  int *x = new (buf, true) int(42);
}
