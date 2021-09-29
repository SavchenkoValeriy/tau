#include "tau/Core/Checker.h"
#include "tau/AIR/AirOps.h"
#include "tau/Core/CheckerPass.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <mlir/Pass/PassManager.h>

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

using namespace tau;
using namespace air;
using namespace core;
using namespace mlir;
using namespace llvm;

namespace {

using MockState = State<0, 0>;

class MockChecker
    : public CheckerWrapper<MockChecker, MockState, ConstantIntOp> {
public:
  StringRef getName() const override { return "Simple test checker"; }
  StringRef getArgument() const override { return "test-checker"; }
  StringRef getDescription() const override { return "Simple test checker"; }

  MAKE_MOCK1(process, void(ConstantIntOp));
};

class CheckerTest {
public:
  void run(const Twine &Snippet) {
    IR = frontend::runClangOnCode(Snippet);
    REQUIRE(IR != nullptr);

    MLIRContext &Context = IR->Context;
    Context.disableMultithreading();
    PassManager PM(&Context);

    SmallVector<Checker *, 1> Checkers{&TestChecker};

    PM.addNestedPass<FuncOp>(createCheckerPass(Checkers));

    REQUIRE(succeeded(PM.run(IR->Module)));
  }

private:
  std::unique_ptr<frontend::Output> IR;

protected:
  MockChecker TestChecker;
};

inline auto constIntOp(int Expected) {
  return trompeloeil::make_matcher<ConstantIntOp>(

      // predicate lambda that checks the condition
      [Expected](ConstantIntOp Op) {
        return Op.getValue().getExtValue() == Expected;
      },

      // print lambda for error message
      [Expected](std::ostream &OS) {
        OS << " matching constIntOp(" << Expected << ")";
      });
}
} // end anonymous namespace

TEST_CASE_METHOD(CheckerTest, "Checker processing interface",
                 "[analysis][checker]") {
  SECTION("No triggers") {
    FORBID_CALL(TestChecker, process(trompeloeil::_));
    run(R"(
void foo() {
}
)");
  }

  SECTION("Trivial trigger") {
    REQUIRE_CALL(TestChecker, process(constIntOp(1)));
    run(R"(
void foo() {
  1;
}
)");
  }

  SECTION("Multiple triggers") {
    REQUIRE_CALL(TestChecker, process(constIntOp(1))).TIMES(5);
    run(R"(
void foo() {
  1;
  1;
  1;
  1;
  1;
}
)");
  }

  SECTION("Triggers on non-trivial CFG") {
    REQUIRE_CALL(TestChecker, process(constIntOp(1)));
    REQUIRE_CALL(TestChecker, process(constIntOp(2)));
    REQUIRE_CALL(TestChecker, process(constIntOp(3)));
    REQUIRE_CALL(TestChecker, process(constIntOp(4)));
    REQUIRE_CALL(TestChecker, process(constIntOp(5)));
    REQUIRE_CALL(TestChecker, process(constIntOp(6)));
    REQUIRE_CALL(TestChecker, process(constIntOp(7)));
    REQUIRE_CALL(TestChecker, process(constIntOp(8)));
    REQUIRE_CALL(TestChecker, process(constIntOp(9)));
    run(R"(
void foo(bool x, bool y, bool z) {
  1;
  if (x) {
    2;
    while (y) {
      3;
      if (z) {
        4;
      } else {
        5;
      }
      6;
    }
    7;
  } else {
    8;
  }
  9;
}
)");
  }
}
