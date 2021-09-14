// RUN: tau-cc -use-of-uninit -verify %s

int test() {
  int a;
  return a; // expected-error{{Use of uninitialized value}}
}
