// RUN: tau-cc -use-of-uninit -verify %s

int test() {
  int a;    // expected-note{{Declared without initial value here}}
  return a; // expected-error{{Use of uninitialized value}}
}

int test_with_value() {
  int a = 42;
  return a; // no error
}
