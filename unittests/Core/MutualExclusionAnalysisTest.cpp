#include "tau/Core/MutualExclusionAnalysis.h"

#include "tau/AIR/AirOps.h"
#include "tau/Core/TopoOrderEnumerator.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassInstrumentation.h>
#include <mlir/Pass/PassManager.h>

#include <catch2/catch_test_macros.hpp>

#include <memory>

using namespace tau;
using namespace core;
using namespace llvm;
using namespace mlir;
using namespace mlir::func;

namespace {

using BlocksByConst = llvm::DenseMap<int, const Block *>;

class BlockCollector
    : public PassWrapper<BlockCollector, OperationPass<FuncOp>> {
public:
  BlockCollector(BlocksByConst &ToFill) : Blocks(ToFill) {}

  StringRef getArgument() const override { return "block-collector"; }
  StringRef getDescription() const override {
    return "Gathers blocks from a function";
  }

  void runOnOperation() override {
    FuncOp Function = getOperation();
    for (Block &BB : Function.getBlocks())
      BB.walk([this, &BB](tau::air::ConstantIntOp ConstInt) {
        int Index = ConstInt.getValue().getExtValue();
        REQUIRE(Blocks.count(Index) == 0);
        Blocks[Index] = &BB;
      });
    markAllAnalysesPreserved();
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlockCollector);

private:
  BlocksByConst &Blocks;
};

class MutualExclusionAnalysisTest {
public:
  MutualExclusionAnalysisTest() : Instrumentor(nullptr), MAM(nullptr) {}

  void run(const Twine &Snippet) {
    IR = frontend::runClangOnCode(Snippet);
    REQUIRE(IR != nullptr);

    MLIRContext &Context = IR->Context;
    Context.disableMultithreading();
    PassManager PM(&Context);

    PM.addNestedPass<FuncOp>(std::make_unique<BlockCollector>(Blocks));

    REQUIRE(succeeded(PM.run(IR->Module)));

    // Find the test function
    FuncOp Func = *IR->Module.getOps<FuncOp>().begin();
    // We need instrumentor to create analysis manager
    Instrumentor = std::make_unique<PassInstrumentor>();
    // AnalysisManager owns all dependant analysis and we need to keep it
    // around so we avoid user after free errors.
    MAM = std::make_unique<ModuleAnalysisManager>(Func, Instrumentor.get());
    AnalysisManager AM = *MAM;
    Analysis = &AM.getAnalysis<MutualExclusionAnalysis>();
  }

  bool areMutuallyExclusive(int Index1, int Index2) {
    return Analysis->areMutuallyExclusive(Blocks[Index1], Blocks[Index2]);
  }

private:
  std::unique_ptr<frontend::Output> IR;
  std::unique_ptr<PassInstrumentor> Instrumentor;
  std::unique_ptr<ModuleAnalysisManager> MAM;
  BlocksByConst Blocks;
  MutualExclusionAnalysis *Analysis = nullptr;
};
} // end anonymous namespace

TEST_CASE_METHOD(MutualExclusionAnalysisTest, "Mutual exclusion for if",
                 "[analysis][mutual-exclusion]") {
  run(R"(
void foo(bool x) {
  1;
  if (x) {
    2;
  } else {
    3;
  }
  4;
}
)");

  CHECK(areMutuallyExclusive(2, 3));
  CHECK_FALSE(areMutuallyExclusive(1, 2));
  CHECK_FALSE(areMutuallyExclusive(1, 3));
  CHECK_FALSE(areMutuallyExclusive(1, 4));
  CHECK_FALSE(areMutuallyExclusive(2, 4));
  CHECK_FALSE(areMutuallyExclusive(3, 4));
}

TEST_CASE_METHOD(MutualExclusionAnalysisTest, "Mutual exclusion for nested if",
                 "[analysis][mutual-exclusion]") {
  run(R"(
void foo(bool x, bool y) {
  1;
  if (x) {
    2;
    if (y) {
      3;
    } else {
      4;
    }
    5;
  } else {
    6;
  }
  7;
}
)");

  CHECK(areMutuallyExclusive(2, 6));
  CHECK(areMutuallyExclusive(3, 4));
  CHECK(areMutuallyExclusive(3, 6));
  CHECK(areMutuallyExclusive(4, 6));
  CHECK_FALSE(areMutuallyExclusive(1, 2));
  CHECK_FALSE(areMutuallyExclusive(1, 3));
  CHECK_FALSE(areMutuallyExclusive(1, 4));
  CHECK_FALSE(areMutuallyExclusive(1, 5));
  CHECK_FALSE(areMutuallyExclusive(1, 6));
  CHECK_FALSE(areMutuallyExclusive(1, 7));
  CHECK_FALSE(areMutuallyExclusive(2, 3));
  CHECK_FALSE(areMutuallyExclusive(2, 4));
  CHECK_FALSE(areMutuallyExclusive(2, 5));
  CHECK_FALSE(areMutuallyExclusive(2, 7));
  CHECK_FALSE(areMutuallyExclusive(3, 5));
  CHECK_FALSE(areMutuallyExclusive(3, 7));
  CHECK_FALSE(areMutuallyExclusive(4, 5));
  CHECK_FALSE(areMutuallyExclusive(4, 7));
  CHECK_FALSE(areMutuallyExclusive(5, 7));
  CHECK_FALSE(areMutuallyExclusive(6, 7));
}

TEST_CASE_METHOD(MutualExclusionAnalysisTest, "Mutual exclusion for loop",
                 "[analysis][mutual-exclusion]") {
  run(R"(
void foo(bool x) {
  1;
  while (x) {
    2;
  }
  3;
}
)");

  CHECK_FALSE(areMutuallyExclusive(1, 2));
  CHECK_FALSE(areMutuallyExclusive(1, 3));
  CHECK_FALSE(areMutuallyExclusive(2, 3));
}

TEST_CASE_METHOD(MutualExclusionAnalysisTest,
                 "Mutual exclusion for nested loops",
                 "[analysis][mutual-exclusion]") {
  run(R"(
void foo(bool x, bool y) {
  1;
  while (x) {
    2;
    while (y) {
      3;
    }
    4;
  }
  5;
}
)");

  CHECK_FALSE(areMutuallyExclusive(1, 2));
  CHECK_FALSE(areMutuallyExclusive(1, 3));
  CHECK_FALSE(areMutuallyExclusive(1, 4));
  CHECK_FALSE(areMutuallyExclusive(1, 5));
  CHECK_FALSE(areMutuallyExclusive(2, 3));
  CHECK_FALSE(areMutuallyExclusive(2, 4));
  CHECK_FALSE(areMutuallyExclusive(2, 5));
  CHECK_FALSE(areMutuallyExclusive(3, 4));
  CHECK_FALSE(areMutuallyExclusive(3, 5));
  CHECK_FALSE(areMutuallyExclusive(4, 5));
}

TEST_CASE_METHOD(MutualExclusionAnalysisTest,
                 "Mutual exclusion for loop with branching",
                 "[analysis][mutual-exclusion]") {
  run(R"(
void foo(bool x, bool y) {
  1;
  while (x) {
    2;
    if (y) {
      3;
    } else {
      4;
    }
    5;
  }
  6;
}
)");

  CHECK_FALSE(areMutuallyExclusive(3, 4));
  CHECK_FALSE(areMutuallyExclusive(1, 2));
  CHECK_FALSE(areMutuallyExclusive(1, 3));
  CHECK_FALSE(areMutuallyExclusive(1, 4));
  CHECK_FALSE(areMutuallyExclusive(1, 5));
  CHECK_FALSE(areMutuallyExclusive(1, 6));
  CHECK_FALSE(areMutuallyExclusive(2, 3));
  CHECK_FALSE(areMutuallyExclusive(2, 4));
  CHECK_FALSE(areMutuallyExclusive(2, 5));
  CHECK_FALSE(areMutuallyExclusive(2, 6));
  CHECK_FALSE(areMutuallyExclusive(3, 5));
  CHECK_FALSE(areMutuallyExclusive(3, 6));
  CHECK_FALSE(areMutuallyExclusive(4, 5));
  CHECK_FALSE(areMutuallyExclusive(4, 6));
  CHECK_FALSE(areMutuallyExclusive(5, 6));
}

TEST_CASE_METHOD(MutualExclusionAnalysisTest,
                 "Mutual exclusion for nested loops with branching",
                 "[analysis][mutual-exclusion]") {
  run(R"(
void foo(bool x, bool y, bool z) {
  1;
  if (z) {
    2;
    while (x) {
      3;
      if (y) {
        4;
        while (z) {
          5;
        }
        6;
      } else {
        7;
      }
      8;
    }
    9;
  } else {
    10;
    while (y) {
      11;
    }
    12;
  }
  13;
}
)");

  CHECK(areMutuallyExclusive(2, 10));
  CHECK(areMutuallyExclusive(3, 10));
  CHECK(areMutuallyExclusive(4, 10));
  CHECK(areMutuallyExclusive(5, 10));
  CHECK(areMutuallyExclusive(6, 10));
  CHECK(areMutuallyExclusive(7, 10));
  CHECK(areMutuallyExclusive(8, 10));
  CHECK(areMutuallyExclusive(9, 10));
  CHECK(areMutuallyExclusive(9, 11));
  CHECK(areMutuallyExclusive(9, 12));

  CHECK_FALSE(areMutuallyExclusive(1, 2));
  CHECK_FALSE(areMutuallyExclusive(1, 3));
  CHECK_FALSE(areMutuallyExclusive(1, 4));
  CHECK_FALSE(areMutuallyExclusive(1, 5));
  CHECK_FALSE(areMutuallyExclusive(1, 6));
  CHECK_FALSE(areMutuallyExclusive(1, 7));
  CHECK_FALSE(areMutuallyExclusive(1, 8));
  CHECK_FALSE(areMutuallyExclusive(1, 9));
  CHECK_FALSE(areMutuallyExclusive(1, 10));
  CHECK_FALSE(areMutuallyExclusive(1, 11));
  CHECK_FALSE(areMutuallyExclusive(1, 12));
  CHECK_FALSE(areMutuallyExclusive(1, 13));

  CHECK_FALSE(areMutuallyExclusive(2, 3));
  CHECK_FALSE(areMutuallyExclusive(2, 4));
  CHECK_FALSE(areMutuallyExclusive(2, 5));
  CHECK_FALSE(areMutuallyExclusive(2, 6));
  CHECK_FALSE(areMutuallyExclusive(2, 7));
  CHECK_FALSE(areMutuallyExclusive(2, 8));
  CHECK_FALSE(areMutuallyExclusive(2, 9));
  CHECK_FALSE(areMutuallyExclusive(2, 13));

  CHECK_FALSE(areMutuallyExclusive(3, 4));
  CHECK_FALSE(areMutuallyExclusive(3, 5));
  CHECK_FALSE(areMutuallyExclusive(3, 6));
  CHECK_FALSE(areMutuallyExclusive(3, 7));
  CHECK_FALSE(areMutuallyExclusive(3, 8));
  CHECK_FALSE(areMutuallyExclusive(3, 9));
  CHECK_FALSE(areMutuallyExclusive(3, 13));

  CHECK_FALSE(areMutuallyExclusive(4, 5));
  CHECK_FALSE(areMutuallyExclusive(4, 6));
  CHECK_FALSE(areMutuallyExclusive(4, 8));
  CHECK_FALSE(areMutuallyExclusive(4, 7));
  CHECK_FALSE(areMutuallyExclusive(4, 9));
  CHECK_FALSE(areMutuallyExclusive(4, 13));

  CHECK_FALSE(areMutuallyExclusive(5, 6));
  CHECK_FALSE(areMutuallyExclusive(5, 7));
  CHECK_FALSE(areMutuallyExclusive(5, 8));
  CHECK_FALSE(areMutuallyExclusive(5, 9));
  CHECK_FALSE(areMutuallyExclusive(5, 13));

  CHECK_FALSE(areMutuallyExclusive(6, 7));
  CHECK_FALSE(areMutuallyExclusive(6, 8));
  CHECK_FALSE(areMutuallyExclusive(6, 9));
  CHECK_FALSE(areMutuallyExclusive(6, 13));

  CHECK_FALSE(areMutuallyExclusive(7, 8));
  CHECK_FALSE(areMutuallyExclusive(7, 9));
  CHECK_FALSE(areMutuallyExclusive(7, 13));

  CHECK_FALSE(areMutuallyExclusive(8, 9));
  CHECK_FALSE(areMutuallyExclusive(8, 13));

  CHECK_FALSE(areMutuallyExclusive(9, 13));

  CHECK_FALSE(areMutuallyExclusive(10, 11));
  CHECK_FALSE(areMutuallyExclusive(10, 12));
  CHECK_FALSE(areMutuallyExclusive(10, 13));

  CHECK_FALSE(areMutuallyExclusive(11, 12));
  CHECK_FALSE(areMutuallyExclusive(11, 13));

  CHECK_FALSE(areMutuallyExclusive(12, 13));
}
