#include "tau/Core/PointsToAnalysis.h"

#include "tau/AIR/AirOps.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <catch2/catch_test_macros.hpp>

using namespace tau;
using namespace core;
using namespace llvm;
using namespace mlir;
using namespace mlir::func;

namespace {
using LineNumber = unsigned;
using ValuesByLineNumber = llvm::DenseMap<LineNumber, mlir::Value>;

class ValueCollector
    : public PassWrapper<ValueCollector, OperationPass<FuncOp>> {
public:
  ValueCollector(ValuesByLineNumber &ToFill,
                 std::unique_ptr<PointsToAnalysis> &Analysis)
      : Values(ToFill), Analysis(Analysis) {}

  StringRef getArgument() const override { return "values-collector"; }
  StringRef getDescription() const override {
    return "Gathers values from a function";
  }

  void runOnOperation() override {
    FuncOp Function = getOperation();
    Analysis =
        std::make_unique<PointsToAnalysis>(getAnalysis<PointsToAnalysis>());

    const auto Collect = [this](Value Candidate) {
      const auto Range = Candidate.getLoc().cast<FusedLoc>();
      if (const auto &Begin =
              Range.getLocations().front().dyn_cast<FileLineColLoc>()) {
        Values[Begin.getLine()] = Candidate;
      }
    };

    Function.walk([&Collect](air::AllocaOp Alloca) { Collect(Alloca); });
    Function.walk([&Collect](air::RefOp Reference) { Collect(Reference); });
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ValueCollector);

private:
  ValuesByLineNumber &Values;
  std::unique_ptr<PointsToAnalysis> &Analysis;
};

class PointsToAnalysisTest {
public:
  void run(const Twine &Snippet) {
    IR = frontend::runClangOnCode(Snippet);
    REQUIRE(IR != nullptr);

    MLIRContext &Context = IR->Context;
    Context.disableMultithreading();
    PassManager PM(&Context);

    PM.addNestedPass<FuncOp>(
        std::make_unique<ValueCollector>(Values, Analysis));

    REQUIRE(succeeded(PM.run(IR->Module)));
  }

  auto getPointsToSet(Value Pointer) const {
    return Analysis->getPointsToSet(Pointer);
  }
  auto getPointsToSet(LineNumber LN) const {
    return getPointsToSet(getValueByLineNumber(LN));
  }

  bool pointsTo(Value Pointer, Value Pointee) const {
    const auto Set = getPointsToSet(Pointer);
    return llvm::find(Set, Pointee) != Set.end();
  }
  bool pointsTo(LineNumber Pointer, LineNumber Pointee) {
    return pointsTo(getValueByLineNumber(Pointer),
                    getValueByLineNumber(Pointee));
  }

  Value getValueByLineNumber(LineNumber LN) const {
    const auto It = Values.find(LN + 1);
    REQUIRE(It != Values.end());
    return It->getSecond();
  }

private:
  std::unique_ptr<frontend::Output> IR;

protected:
  ValuesByLineNumber Values;
  std::unique_ptr<PointsToAnalysis> Analysis;
};
} // end anonymous namespace

TEST_CASE_METHOD(PointsToAnalysisTest, "Trivial aliasing",
                 "[analysis][alias][points-to]") {
  run(R"(
void foo() {
  int x = 1;
  int *y = &x;
}
)");
  CHECK(getPointsToSet(2).empty());
  CHECK(pointsTo(3, 2));
}

TEST_CASE_METHOD(PointsToAnalysisTest, "Multiple escapes",
                 "[analysis][alias][points-to]") {
  run(R"(
void foo() {
  int x = 1;
  int y = 1;
  int *z = &y;
  int *w = &x;
}
)");
  CHECK(pointsTo(4, 3));
  CHECK(pointsTo(5, 2));
  CHECK_FALSE(pointsTo(3, 2));
  // FIXME: right now we overapproximate the sizes of points-to sets
  CHECK(pointsTo(4, 2));
  CHECK(pointsTo(5, 3));
}

TEST_CASE_METHOD(PointsToAnalysisTest, "Type noalising",
                 "[analysis][alias][points-to]") {
  run(R"(
void foo() {
  int x = 1;
  float y = 36.6;
  float *z = &y;
  int *w = &x;
}
)");
  CHECK(pointsTo(4, 3));
  CHECK(pointsTo(5, 2));
  CHECK_FALSE(pointsTo(3, 2));
  CHECK_FALSE(pointsTo(4, 2));
  CHECK_FALSE(pointsTo(5, 3));
}
