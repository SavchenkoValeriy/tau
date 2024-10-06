// RUN: tau-cc -use-after-free -verify %s -- -std=c++17

int test_simple_if(bool cond) {
  int *x = new int(42); // expected-note{{Allocated here}}
  int c = 42;
  int *a = &c;
  if (cond) {
  } else {
    a = x;
    delete x; // expected-note{{Deallocated here}}
  }
  return *a; // expected-error{{Use of deallocated pointer}}
}

int test_exclusive_if(bool cond) {
  int *x = new int(42);
  int c = 42;
  int *a = &c;
  if (cond) {
    a = x;
  } else {
    delete x;
  }
  return *a;
}
