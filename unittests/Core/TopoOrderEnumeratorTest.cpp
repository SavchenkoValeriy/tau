#include "tau/Core/TopoOrderEnumerator.h"

#include "tau/AIR/AirOps.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <catch2/catch.hpp>

using namespace tau;
using namespace core;
using namespace llvm;
using namespace mlir;

namespace {

using BlocksByConst = llvm::DenseMap<int, const Block *>;

class BlockCollector
    : public PassWrapper<BlockCollector, OperationPass<FuncOp>> {
public:
  BlockCollector(BlocksByConst &ToFill,
                 std::unique_ptr<TopoOrderBlockEnumerator> &Enumerator)
      : Blocks(ToFill), Enumerator(Enumerator) {}

  StringRef getArgument() const override { return "block-collector"; }
  StringRef getDescription() const override {
    return "Gathers blocks from a function";
  }

  void runOnOperation() override {
    FuncOp Function = getOperation();
    Enumerator = std::make_unique<TopoOrderBlockEnumerator>(
        getAnalysis<TopoOrderBlockEnumerator>());
    for (Block &BB : Function.getBlocks())
      BB.walk([this, &BB](tau::air::ConstantIntOp ConstInt) {
        int Index = ConstInt.getValue().getExtValue();
        REQUIRE(Blocks.count(Index) == 0);
        Blocks[Index] = &BB;
      });
  }

private:
  BlocksByConst &Blocks;
  std::unique_ptr<TopoOrderBlockEnumerator> &Enumerator;
};

class TopoOrderEnumeratorTest {
public:
  void run(const Twine &Snippet) {
    IR = frontend::runClangOnCode(Snippet);
    REQUIRE(IR != nullptr);

    MLIRContext &Context = IR->Context;
    Context.disableMultithreading();
    PassManager PM(&Context);

    PM.addNestedPass<FuncOp>(
        std::make_unique<BlockCollector>(Blocks, Enumerator));

    REQUIRE(succeeded(PM.run(IR->Module)));
  }

  int getOrder(int Index) {
    return Enumerator->getTopoOrderIndex(Blocks[Index]);
  }

private:
  std::unique_ptr<frontend::Output> IR;

protected:
  BlocksByConst Blocks;
  std::unique_ptr<TopoOrderBlockEnumerator> Enumerator;
};
} // end anonymous namespace

TEST_CASE_METHOD(TopoOrderEnumeratorTest, "Order for if", "[analysis][order]") {
  run(R"(
void foo(bool x) {
  1;
  if (x) {
    2;
  }
  3;
}
)");

  CHECK(getOrder(1) > getOrder(2));
  CHECK(getOrder(2) > getOrder(3));
  CHECK(getOrder(1) > getOrder(3));
}

TEST_CASE_METHOD(TopoOrderEnumeratorTest, "Order for if-else",
                 "[analysis][order]") {
  run(R"(
void foo(bool x) {
  1;
  if (x) {
    2;
  } else {
    4;
  }
  3;
}
)");

  CHECK(getOrder(1) > getOrder(2));
  CHECK(getOrder(2) > getOrder(3));
  CHECK(getOrder(1) > getOrder(3));
  CHECK(getOrder(1) > getOrder(4));
  CHECK(getOrder(1) > getOrder(3));
}

TEST_CASE_METHOD(TopoOrderEnumeratorTest, "Order for nested if-else",
                 "[analysis][order]") {
  run(R"(
void foo(bool x, bool y) {
  1;
  if (x) {
    2;
    if (y) {
      5;
    } else {
      6;
    }
  } else if (y) {
    4;
  } else {
    7;
  }
  3;
}
)");

  CHECK(getOrder(1) > getOrder(2));

  CHECK(getOrder(2) > getOrder(3));
  CHECK(getOrder(2) > getOrder(5));
  CHECK(getOrder(2) > getOrder(6));

  CHECK(getOrder(1) > getOrder(3));
  CHECK(getOrder(1) > getOrder(4));
  CHECK(getOrder(1) > getOrder(7));

  CHECK(getOrder(4) > getOrder(3));
  CHECK(getOrder(7) > getOrder(3));

  CHECK(getOrder(1) > getOrder(3));
}

TEST_CASE_METHOD(TopoOrderEnumeratorTest, "Order for loop",
                 "[analysis][order]") {
  run(R"(
void foo(bool x) {
  1;
  while (x) {
    2;
  }
  3;
}
)");

  CHECK(getOrder(1) > getOrder(2));
  CHECK(getOrder(2) > getOrder(3));
  CHECK(getOrder(1) > getOrder(3));
}

TEST_CASE_METHOD(TopoOrderEnumeratorTest, "Order for if in loop",
                 "[analysis][order][!shouldfail]") {
  run(R"(
void foo(bool x, bool y) {
  1;
  while (x) {
    2;
    if (y) {
      4;
    } else {
      5;
    }
    6;
  }
  3;
}
)");

  CHECK(getOrder(1) > getOrder(2));

  CHECK(getOrder(2) > getOrder(3));
  CHECK(getOrder(2) > getOrder(4));
  CHECK(getOrder(2) > getOrder(5));
  CHECK(getOrder(2) > getOrder(6));

  CHECK(getOrder(1) > getOrder(3));
  CHECK(getOrder(1) > getOrder(4));
  CHECK(getOrder(1) > getOrder(5));

  CHECK(getOrder(4) > getOrder(3));
  CHECK(getOrder(4) > getOrder(6));

  CHECK(getOrder(5) > getOrder(3));
  CHECK(getOrder(5) > getOrder(6));

  CHECK(getOrder(6) > getOrder(3));
}

TEST_CASE_METHOD(TopoOrderEnumeratorTest, "Order for nested loops",
                 "[analysis][order][!shouldfail]") {
  run(R"(
void foo(bool x, bool y) {
  1;
  while (x) {
    2;
    if (y) {
      4;
    } else {
      while (x) {
        5;
        if (y) {
          7;
        } else {
          8;
        }
        9;
      }
      10;
    }
    6;
  }
  3;
}
)");

  CHECK(getOrder(1) > getOrder(2));

  CHECK(getOrder(2) > getOrder(3));
  CHECK(getOrder(2) > getOrder(4));
  CHECK(getOrder(2) > getOrder(5));
  CHECK(getOrder(2) > getOrder(6));
  CHECK(getOrder(2) > getOrder(7));
  CHECK(getOrder(2) > getOrder(8));
  CHECK(getOrder(2) > getOrder(9));
  CHECK(getOrder(2) > getOrder(10));

  CHECK(getOrder(1) > getOrder(3));
  CHECK(getOrder(1) > getOrder(4));
  CHECK(getOrder(1) > getOrder(5));

  CHECK(getOrder(4) > getOrder(3));
  CHECK(getOrder(4) > getOrder(5));
  CHECK(getOrder(4) > getOrder(6));
  CHECK(getOrder(4) > getOrder(7));
  CHECK(getOrder(4) > getOrder(8));
  CHECK(getOrder(4) > getOrder(9));
  CHECK(getOrder(4) > getOrder(10));

  CHECK(getOrder(5) > getOrder(3));
  CHECK(getOrder(5) > getOrder(6));
  CHECK(getOrder(5) > getOrder(7));
  CHECK(getOrder(5) > getOrder(8));

  CHECK(getOrder(6) > getOrder(3));

  CHECK(getOrder(7) > getOrder(9));
  CHECK(getOrder(8) > getOrder(9));
}
