#include "tau/Core/FlowSensitive.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerPass.h"
#include "tau/Core/State.h"
#include "tau/Core/StateEventForest.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Casting.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <initializer_list>
#include <iterator>
#include <memory>

#include <catch2/catch.hpp>

using namespace tau;
using namespace core;
using namespace mlir;
using namespace mlir::func;
using namespace llvm;

namespace {

using SimpleCheckerState = State<2, 1>;
constexpr SimpleCheckerState FOO = SimpleCheckerState::getNonErrorState(0);
constexpr SimpleCheckerState BAR = SimpleCheckerState::getNonErrorState(1);
constexpr SimpleCheckerState FOOBAR = SimpleCheckerState::getErrorState(0);

class SimpleChecker
    : public CheckerBase<SimpleChecker, SimpleCheckerState, CallOp> {
public:
  StringRef getName() const override { return "Simple test checker"; }
  StringRef getArgument() const override { return "test-checker"; }
  StringRef getDescription() const override { return "Simple test checker"; }

  void process(CallOp Call) {
    StringRef CalledFunction = Call.getCallee();
    if (int KnownFunc = llvm::StringSwitch<int>(CalledFunction)
                            .StartsWith("void foobar", 3)
                            .StartsWith("void bar", 2)
                            .StartsWith("void foo", 1)
                            .Default(0))
      switch (KnownFunc) {
      case 1:
        markChange(Call.getOperation(), Call.getOperand(0), FOO);
        break;
      case 2:
        markChange(Call.getOperation(), Call.getOperand(0), FOO, BAR);
        break;
      case 3:
        markChange(Call.getOperation(), Call.getOperand(0), BAR, FOOBAR);
        break;
      }
  }
};

class FlowSensitiveIssuesHarvester
    : public PassWrapper<FlowSensitiveIssuesHarvester, OperationPass<FuncOp>> {
public:
  FlowSensitiveIssuesHarvester(
      std::vector<FlowSensitiveAnalysis::Issue> &IssuesToFill,
      StateEventForest &Forest)
      : FoundIssues(IssuesToFill), Forest(Forest) {}

  StringRef getArgument() const override { return "flow-sen-checker"; }
  StringRef getDescription() const override {
    return "Gathers found flow-sensitive issues";
  }

  void runOnOperation() override {
    auto &FlowSen = getAnalysis<FlowSensitiveAnalysis>();

    // We need to make sure that we own the issues.
    llvm::copy(FlowSen.getFoundIssues(), std::back_inserter(FoundIssues));

    // We also need to keep forest alive, so that events are also alive
    // when the analysis is gone.
    Forest = std::move(FlowSen.getEventForest());
    markAllAnalysesPreserved();
  }

private:
  std::vector<FlowSensitiveAnalysis::Issue> &FoundIssues;
  StateEventForest &Forest;
};

class FlowSensitiveAnalysisTest {
public:
  template <class CheckerT> void run(const Twine &Snippet) {
    IR = frontend::runClangOnCode(Snippet);
    REQUIRE(IR != nullptr);

    MLIRContext &Context = IR->Context;
    Context.disableMultithreading();
    PassManager PM(&Context);

    CheckerT TestChecker;
    SmallVector<AbstractChecker *, 1> Checkers{&TestChecker};

    PM.addNestedPass<FuncOp>(createCheckerPass(Checkers));
    PM.addNestedPass<FuncOp>(std::make_unique<FlowSensitiveIssuesHarvester>(
        FoundIssues, EventForest));

    REQUIRE(succeeded(PM.run(IR->Module)));
  }

private:
  std::unique_ptr<frontend::Output> IR;
  StateEventForest EventForest;

protected:
  std::vector<FlowSensitiveAnalysis::Issue> FoundIssues;
};
} // end anonymous namespace

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Trivial sequence of events",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x) {
  foo(x);
  bar(x);
  foobar(x);
}
)");
  REQUIRE(FoundIssues.size() == 1);
  auto Issue = FoundIssues[0];
  CHECK(Issue.Guaranteed);

  auto Foobar = dyn_cast_or_null<CallOp>(Issue.ErrorEvent.Location);
  REQUIRE(Foobar);
  CHECK(Foobar.getCallee().startswith("void foobar"));

  REQUIRE(Issue.ErrorEvent.Parent != nullptr);
  auto BarEvent = *Issue.ErrorEvent.Parent;
  auto Bar = dyn_cast_or_null<CallOp>(BarEvent.Location);
  REQUIRE(Bar);
  CHECK(Bar.getCallee().startswith("void bar"));

  REQUIRE(BarEvent.Parent != nullptr);
  auto FooEvent = *BarEvent.Parent;
  auto Foo = dyn_cast_or_null<CallOp>(FooEvent.Location);
  REQUIRE(Foo);
  CHECK(Foo.getCallee().startswith("void foo"));

  CHECK(FooEvent.Parent == nullptr);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest,
                 "Events always following one another", "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  foo(x);

  if (x > 0) {
    z = x + y;
  } else if (y < 10) {
    z = 42;
  } else {
    z = x;
  }

  bar(x);

  if (z + 10 == y)
    z++;

  foobar(x);
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Sequential events in a branch",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  if (y > 0) {
    foo(x);
    bar(x);
    foobar(x);
  }
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Sequential events in a loop",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  while (x > 0) {
    foo(x);
    bar(x);
    foobar(x);
  }
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Events domintating one another",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int z) {
  if (z > 0) {
    foo(x);
    if (y > 0) {
      bar(x);
      if (z > 0) {
        foobar(x);
      }
    }
  }
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest,
                 "Events post-domintating one another", "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int z) {
  if (z > 0) {
    if (y > 0) {
      if (z > 0) {
        foo(x);
      }
      bar(x);
    }
    foobar(x);
  }
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Conditional event #1",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  if (y > 0) {
    foo(x);
  }
  bar(x);
  foobar(x);
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Conditional event #2",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  foo(x);
  if (y > 0) {
    bar(x);
  }
  foobar(x);
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Conditional event #3",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  foo(x);
  bar(x);
  if (y > 0) {
    foobar(x);
  }
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Conditional event #4",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  if (y > 0) {
    foo(x);
    bar(x);
  }
  foobar(x);
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Conditional event #5",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  foo(x);
  if (y > 0) {
    bar(x);
    foobar(x);
  }
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Mutually exclusive events",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  if (x > 0)
    foo(x);
  else
    bar(x);
  foobar(x);
}
)");

  REQUIRE(FoundIssues.size() == 0);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest,
                 "Potentially mutually exclusive events",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  if (x > 0)
    foo(x);
  bar(x);
  if (y < 0)
    foobar(x);
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(!FoundIssues[0].Guaranteed);
}

TEST_CASE_METHOD(FlowSensitiveAnalysisTest, "Error event inside of a loop",
                 "[analysis][flowsen]") {
  run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x, int y, int &z) {
  foo(x);
  bar(x);
  while (x < 0)
    foobar(x);
}
)");

  REQUIRE(FoundIssues.size() == 1);
  CHECK(FoundIssues[0].Guaranteed);
}
