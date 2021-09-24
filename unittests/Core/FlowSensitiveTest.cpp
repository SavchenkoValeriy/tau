#include "tau/Core/FlowSensitive.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerPass.h"
#include "tau/Core/State.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <clang/Tooling/Tooling.h>
#include <initializer_list>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <memory>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <gtest/gtest.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

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
      ArrayRef<FlowSensitiveAnalysis::Issue> &IssuesToFill)
      : FoundIssues(IssuesToFill) {}

  StringRef getArgument() const override { return "flow-sen-checker"; }
  StringRef getDescription() const override {
    return "Gathers found flow-sensitive issues";
  }

  void runOnOperation() override {
    auto &FlowSen = getAnalysis<FlowSensitiveAnalysis>();
    FoundIssues = FlowSen.getFoundIssues();
  }

  ArrayRef<FlowSensitiveAnalysis::Issue> getFoundIssues() const {
    return FoundIssues;
  }

private:
  ArrayRef<FlowSensitiveAnalysis::Issue> &FoundIssues;
};

class FlowSensitiveAnalysisTest : public ::testing::Test {
public:
  template <class CheckerT>
  ArrayRef<FlowSensitiveAnalysis::Issue> run(const Twine &Snippet) {
    auto IR = frontend::runClangOnCode(Snippet);
    MLIRContext &Context = IR->Context;
    Context.printOpOnDiagnostic(false);

    PassManager PM(&Context);
    CheckerT TestChecker;
    SmallVector<Checker *, 1> Checkers{&TestChecker};
    PM.addNestedPass<FuncOp>(createCheckerPass(Checkers));
    ArrayRef<FlowSensitiveAnalysis::Issue> FoundIssues;
    PM.addNestedPass<FuncOp>(
        std::make_unique<FlowSensitiveIssuesHarvester>(FoundIssues));
    EXPECT_TRUE(succeeded(PM.run(IR->Module)));

    return FoundIssues;
  }
};
} // end anonymous namespace

TEST_F(FlowSensitiveAnalysisTest, Basic) {
  auto Issues = run<SimpleChecker>(R"(
void foobar(int &x) {}
void foo(int &x) {}
void bar(int &x) {}

void test(int x) {
  foo(x);
  bar(x);
  foobar(x);
}
)");
  ASSERT_EQ(Issues.size(), 1);
}
