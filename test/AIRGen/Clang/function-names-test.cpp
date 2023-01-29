// RUN: tau-cc -dump=air %s > %t.out 2>&1
// RUN: FileCheck %s < %t.out

namespace a {
namespace b {
class A { int x; float y; };
// CHECK: air.def @"a::b::A" : !air.rec<><x : si32, y : f32>

template <class T, class U>
class B {};
// TODO: generate defintion for B as well
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
// CHECK: air.def @"c::C<int>::D<double>" : !air.rec<>
// CHECK: func.func @"void c::C<int>::D<double>::foo(a::b::A, a::b::B<double, int>)"
template class c::C<bool>::D<a::b::A>;
// CHECK: air.def @"c::C<bool>::D<a::b::A>" : !air.rec<>
// CHECK: func.func @"void c::C<bool>::D<a::b::A>::foo(a::b::A, a::b::B<a::b::A, bool>)"
