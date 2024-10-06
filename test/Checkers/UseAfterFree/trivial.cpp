// RUN: tau-cc -use-after-free -verify %s -- -std=c++17

struct A {
  int *x;
};

int test() {
  int *x;
  {
    int a = 42; // expected-note{{Allocated here}}
    x = &a;
  } // expected-note{{Deallocated here}}
  return *x; // expected-error{{Use of deallocated pointer}}
}

int test_field() {
  A b;
  b.x = 0;
  {
    int a = 42; // expected-note{{Allocated here}}
    b.x = &a;
  } // expected-note{{Deallocated here}}
  return *b.x; // expected-error{{Use of deallocated pointer}}
}

int test_new_delete() {
  int *x = new int(42); // expected-note{{Allocated here}}
  delete x; // expected-note{{Deallocated here}}
  return *x; // expected-error{{Use of deallocated pointer}}
}

int test_new_delete_field() {
  A b;
  b.x = new int(42); // expected-note{{Allocated here}}
  delete b.x; // expected-note{{Deallocated here}}
  return *b.x; // expected-error{{Use of deallocated pointer}}
}

int test_new_delete_with_extra_var() {
  int *x = new int(42); // expected-note{{Allocated here}}
  int c = 42;
  int *a = &c;
  a = x;
  delete x; // expected-note{{Deallocated here}}
  return *a; // expected-error{{Use of deallocated pointer}}
}
