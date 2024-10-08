#include "tau/Core/FlowSensitive.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerPass.h"
#include "tau/Core/Events.h"
#include "tau/Core/State.h"
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

#include <catch2/catch_test_macros.hpp>

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
consteval auto makeStateMachine() {
  StateMachine<SimpleCheckerState> SM;
  SM.markInitial(FOO);
  SM.addEdge(FOO, BAR);
  SM.addEdge(BAR, FOOBAR);
  return SM;
}
constexpr auto SimpleCheckerStateMachine = makeStateMachine();

class SimpleChecker : public CheckerBase<SimpleChecker, SimpleCheckerState,
                                         SimpleCheckerStateMachine, CallOp> {
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
        markChange<FOO>(Call.getOperation(), Call.getOperand(0));
        break;
      case 2:
        markChange<FOO, BAR>(Call.getOperation(), Call.getOperand(0));
        break;
      case 3:
        markChange<BAR, FOOBAR>(Call.getOperation(), Call.getOperand(0));
        break;
      }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimpleChecker);
};

class FlowSensitiveIssuesHarvester
    : public PassWrapper<FlowSensitiveIssuesHarvester, OperationPass<FuncOp>> {
public:
  FlowSensitiveIssuesHarvester(
      std::vector<FlowSensitiveAnalysis::Issue> &IssuesToFill,
      EventHierarchy &Hierarchy)
      : FoundIssues(IssuesToFill), Hierarchy(Hierarchy) {}

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
    Hierarchy = std::move(FlowSen.getEventHierarchy());
    markAllAnalysesPreserved();
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlowSensitiveIssuesHarvester);

private:
  std::vector<FlowSensitiveAnalysis::Issue> &FoundIssues;
  EventHierarchy &Hierarchy;
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
    PM.addNestedPass<FuncOp>(
        std::make_unique<FlowSensitiveIssuesHarvester>(FoundIssues, Hierarchy));

    REQUIRE(succeeded(PM.run(IR->Module)));
  }

private:
  std::unique_ptr<frontend::Output> IR;
  EventHierarchy Hierarchy;

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
  REQUIRE(Issue.Events.size() == 3);

  REQUIRE(Issue.Events.front().is<const StateEvent *>());
  auto Foobar = dyn_cast_or_null<CallOp>(
      Issue.Events.front().get<const StateEvent *>()->getLocation());
  REQUIRE(Foobar);
  CHECK(Foobar.getCallee().starts_with("void foobar"));

  REQUIRE(Issue.Events[1].is<const StateEvent *>());
  auto &BarEvent = *Issue.Events[1].get<const StateEvent *>();
  auto Bar = dyn_cast_or_null<CallOp>(BarEvent.getLocation());
  REQUIRE(Bar);
  CHECK(Bar.getCallee().starts_with("void bar"));

  REQUIRE(Issue.Events[2].is<const StateEvent *>());
  auto &FooEvent = *Issue.Events[2].get<const StateEvent *>();
  auto Foo = dyn_cast_or_null<CallOp>(FooEvent.getLocation());
  REQUIRE(Foo);
  CHECK(Foo.getCallee().starts_with("void foo"));

  CHECK(FooEvent.getParents().empty());
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
