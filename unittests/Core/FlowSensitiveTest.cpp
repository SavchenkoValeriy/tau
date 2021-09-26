#include "tau/Core/FlowSensitive.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerPass.h"
#include "tau/Core/State.h"
#include "tau/Core/StateEventForest.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <clang/Tooling/Tooling.h>
#include <initializer_list>
#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Casting.h>
#include <memory>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <gtest/gtest.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

using namespace tau;
using namespace core;
using namespace mlir;
using namespace llvm;

namespace {

using SimpleCheckerState = State<2, 1>;
constexpr SimpleCheckerState FOO = SimpleCheckerState::getNonErrorState(0);
constexpr SimpleCheckerState BAR = SimpleCheckerState::getNonErrorState(1);
constexpr SimpleCheckerState FOOBAR = SimpleCheckerState::getErrorState(0);

class SimpleChecker
    : public CheckerWrapper<SimpleChecker, SimpleCheckerState, mlir::CallOp> {
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

class FlowSensitiveAnalysisTest : public ::testing::Test {
public:
  template <class CheckerT> void run(const Twine &Snippet) {
    IR = frontend::runClangOnCode(Snippet);
    ASSERT_NE(IR, nullptr);

    MLIRContext &Context = IR->Context;
    PassManager PM(&Context);

    CheckerT TestChecker;
    SmallVector<Checker *, 1> Checkers{&TestChecker};

    PM.addNestedPass<FuncOp>(createCheckerPass(Checkers));
    PM.addNestedPass<FuncOp>(std::make_unique<FlowSensitiveIssuesHarvester>(
        FoundIssues, EventForest));

    ASSERT_TRUE(succeeded(PM.run(IR->Module)));
  }

private:
  std::unique_ptr<frontend::Output> IR;
  StateEventForest EventForest;

protected:
  std::vector<FlowSensitiveAnalysis::Issue> FoundIssues;
};
} // end anonymous namespace

TEST_F(FlowSensitiveAnalysisTest, Basic) {
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
  ASSERT_EQ(FoundIssues.size(), 1);
  auto Issue = FoundIssues[0];
  EXPECT_TRUE(Issue.Guaranteed);

  auto Foobar = dyn_cast_or_null<CallOp>(Issue.ErrorEvent.Location);
  ASSERT_TRUE(Foobar);
  EXPECT_TRUE(Foobar.callee().startswith("void foobar"));

  ASSERT_NE(Issue.ErrorEvent.Parent, nullptr);
  auto BarEvent = *Issue.ErrorEvent.Parent;
  auto Bar = dyn_cast_or_null<CallOp>(BarEvent.Location);
  ASSERT_TRUE(Bar);
  EXPECT_TRUE(Bar.callee().startswith("void bar"));

  ASSERT_NE(BarEvent.Parent, nullptr);
  auto FooEvent = *BarEvent.Parent;
  auto Foo = dyn_cast_or_null<CallOp>(FooEvent.Location);
  ASSERT_TRUE(Foo);
  EXPECT_TRUE(Foo.callee().startswith("void foo"));

  ASSERT_EQ(FooEvent.Parent, nullptr);
}
