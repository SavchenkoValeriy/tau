#include "tau/Core/AddressTakenAnalysis.h"

#include "tau/AIR/AirOps.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <catch2/catch_test_macros.hpp>

using namespace tau;
using namespace core;
using namespace llvm;
using namespace mlir;
using namespace mlir::func;

namespace {

using AllocasById = llvm::DenseMap<int, mlir::Value>;

class AllocaCollector
    : public PassWrapper<AllocaCollector, OperationPass<FuncOp>> {
public:
  AllocaCollector(AllocasById &ToFill,
                  std::unique_ptr<AddressTakenAnalysis> &Analysis)
      : Allocas(ToFill), Analysis(Analysis) {}

  StringRef getArgument() const override { return "block-collector"; }
  StringRef getDescription() const override {
    return "Gathers blocks from a function";
  }

  void runOnOperation() override {
    FuncOp Function = getOperation();
    Analysis = std::make_unique<AddressTakenAnalysis>(
        getAnalysis<AddressTakenAnalysis>());
    Function.walk([this](air::StoreOp Store) {
      if (!isa<air::AllocaOp>(Store.getAddress().getDefiningOp()))
        return;

      if (auto ConstInt =
              dyn_cast<air::ConstantIntOp>(Store.getValue().getDefiningOp())) {
        int Id = ConstInt.getValue().getExtValue();
        REQUIRE(Allocas.count(Id) == 0);
        Allocas[Id] = Store.getAddress();
      }
    });
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AllocaCollector);

private:
  AllocasById &Allocas;
  std::unique_ptr<AddressTakenAnalysis> &Analysis;
};

class AddressTakenAnalysisTest {
public:
  void run(const Twine &Snippet) {
    IR = frontend::runClangOnCode(Snippet);
    REQUIRE(IR != nullptr);

    MLIRContext &Context = IR->Context;
    Context.disableMultithreading();
    PassManager PM(&Context);

    PM.addNestedPass<FuncOp>(
        std::make_unique<AllocaCollector>(Allocas, Analysis));

    REQUIRE(succeeded(PM.run(IR->Module)));
  }

  int hasAddressBeenTaken(int Id) {
    return Analysis->hasAddressBeenTaken(Allocas[Id]);
  }

private:
  std::unique_ptr<frontend::Output> IR;

protected:
  AllocasById Allocas;
  std::unique_ptr<AddressTakenAnalysis> Analysis;
};
} // end anonymous namespace

TEST_CASE_METHOD(AddressTakenAnalysisTest, "Address trivially not taken",
                 "[analysis][alias][address]") {
  run(R"(
int foo() {
  int x = 1;
  return x;
}
)");

  CHECK_FALSE(hasAddressBeenTaken(1));
}

TEST_CASE_METHOD(AddressTakenAnalysisTest, "Address captured by a pointer",
                 "[analysis][alias][address]") {
  run(R"(
int foo() {
  int x = 1;
  int *y = &x;
  return x;
}
)");

  CHECK(hasAddressBeenTaken(1));
}

TEST_CASE_METHOD(AddressTakenAnalysisTest, "Address captured by a reference",
                 "[analysis][alias][address]") {
  run(R"(
int foo() {
  int x = 1;
  int &y = x;
  return x;
}
)");

  CHECK(hasAddressBeenTaken(1));
}

TEST_CASE_METHOD(AddressTakenAnalysisTest, "Address captured by a function",
                 "[analysis][alias][address]") {
  run(R"(
void bar(int &x);
int foo() {
  int x = 1;
  bar(x);
  return x;
}
)");

  CHECK(hasAddressBeenTaken(1));
}

TEST_CASE_METHOD(AddressTakenAnalysisTest, "Address not taken complex",
                 "[analysis][alias][address]") {
  run(R"(
int bar(int x);
int foo() {
  int x = 1;
  int y = 2;
  int z = bar(x) * y;
  return z + x * y;
}
)");

  CHECK_FALSE(hasAddressBeenTaken(1));
  CHECK_FALSE(hasAddressBeenTaken(2));
}
