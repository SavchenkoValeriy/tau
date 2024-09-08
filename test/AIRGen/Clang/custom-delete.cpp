// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

struct DefaultDelete {
  int x;
  ~DefaultDelete() {}
};

struct CustomDelete {
  int y;
  ~CustomDelete() {}

  static void operator delete(void* ptr) {
    // Custom delete implementation
  }
};

void test_default_delete() {
  DefaultDelete* obj = new DefaultDelete();
  delete obj;
}

// CHECK-LABEL: @"void test_default_delete()"
// CHECK: %[[OBJ:.*]] = air.alloca : !air<ptr !air<ptr !air<recref @DefaultDelete>>>
// CHECK: %[[NEW:.*]] = air.halloca : !air<ptr !air<recref @DefaultDelete>>
// CHECK: call @"void DefaultDelete::DefaultDelete(DefaultDelete *)"(%[[NEW]])
// CHECK: air.store %[[NEW]] -> %[[OBJ]]
// CHECK: %[[LOAD:.*]] = air.load %[[OBJ]]
// CHECK: call @"void DefaultDelete::~DefaultDelete(DefaultDelete *)"(%[[LOAD]])
// CHECK: air.hdealloca %[[LOAD]] : !air<ptr !air<recref @DefaultDelete>>
// CHECK: air.dealloca %[[OBJ]]

void test_custom_delete() {
  CustomDelete* obj = new CustomDelete();
  delete obj;
}

// CHECK-LABEL: @"void test_custom_delete()"
// CHECK: %[[OBJ:.*]] = air.alloca : !air<ptr !air<ptr !air<recref @CustomDelete>>>
// CHECK: %[[NEW:.*]] = air.halloca : !air<ptr !air<recref @CustomDelete>>
// CHECK: call @"void CustomDelete::CustomDelete(CustomDelete *)"(%[[NEW]])
// CHECK: air.store %[[NEW]] -> %[[OBJ]]
// CHECK: %[[LOAD:.*]] = air.load %[[OBJ]]
// CHECK: call @"void CustomDelete::~CustomDelete(CustomDelete *)"(%[[LOAD]])
// CHECK: %[[CAST:.*]] = air.bitcast %[[LOAD]] : !air<ptr !air<recref @CustomDelete>> to !air<ptr !air.void>
// CHECK: call @"void CustomDelete::operator delete(void *)"(%[[CAST]])
// CHECK: air.dealloca %[[OBJ]]
