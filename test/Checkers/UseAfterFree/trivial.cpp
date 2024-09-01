// RUN: tau-cc -use-after-free -verify %s -- -std=c++17

struct A {
  int *x;
};

int test() {
  int *x;
  {
    int a = 42;
    x = &a;
  } // expected-note{{Deallocated here}}
  return *x; // expected-error{{Use of deallocated pointer}}
}

int test_field() {
  A b;
  b.x = 0;
  {
    int a = 42;
    b.x = &a;
  } // expected-note{{Deallocated here}}
  return *b.x; // expected-error{{Use of deallocated pointer}}
}
