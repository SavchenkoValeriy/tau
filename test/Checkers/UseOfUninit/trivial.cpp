// RUN: tau-cc -use-of-uninit -verify %s

int test() {
  int a;    // expected-note{{Declared without initial value here}}
  return a; // expected-error{{Use of uninitialized value}}
}

int test_with_value() {
  int a = 42;
  return a; // no error
}

int test_reassign() {
  int a;
  a = 42;
  return a; // no error
}

int test_reassign_if_error(bool cond) {
  int a; // expected-note{{Declared without initial value here}}
  if (cond)
    a = 42;
  return a; // expected-error{{Use of uninitialized value}}
}

int test_reassign_if_noerror(bool cond) {
  int a;
  if (cond)
    a = 42;
  else
    a = 15;
  return a; // no error
}
