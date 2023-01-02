// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

namespace a {
namespace b {
class A {};

template <class T, class U>
class B {};
} // end namespace b
} // end namespace a

using namespace a;
using namespace b;

namespace c {
template <class T>
struct C {
  template <class U>
  struct D {
    static void foo(A param1, B<U, T> param2) {}
  };
};
} // end namespace c

template class c::C<int>::D<double>;
// CHECK: func.func @"static void c::C<int>::D<double>::foo(a::b::A param1, a::b::B<double, int> param2)"
template class c::C<bool>::D<a::b::A>;
// CHECK: func.func @"static void c::C<bool>::D<a::b::A>::foo(a::b::A param1, a::b::B<a::b::A, bool> param2)"
