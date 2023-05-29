#include "tau/Core/ReachingDefs.h"

#include "tau/AIR/AirOps.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <clang/Tooling/Tooling.h>
#include <iterator>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

namespace mlir {
std::ostream &operator<<(std::ostream &OS, const Value &V) {
  std::string Buffer;
  llvm::raw_string_ostream SS(Buffer);
  V.getDefiningOp()->print(SS);
  return OS << SS.str();
}
} // end namespace mlir

#include <catch2/catch.hpp>

using namespace tau;
using namespace core;
using namespace llvm;
using namespace mlir;
using namespace mlir::func;

namespace {
using LineNumber = unsigned;
using OpsByLineNumber =
    llvm::DenseMap<LineNumber, std::vector<mlir::Operation *>>;

class OpCollector : public PassWrapper<OpCollector, OperationPass<FuncOp>> {
public:
  OpCollector(OpsByLineNumber &ToFill, std::unique_ptr<ReachingDefs> &Analysis)
      : Ops(ToFill), Analysis(Analysis) {}

  StringRef getArgument() const override { return "op-collector"; }
  StringRef getDescription() const override {
    return "Gathers op lines from a function";
  }

  void runOnOperation() override {
    FuncOp Function = getOperation();
    Analysis =
        std::make_unique<ReachingDefs>(std::move(getAnalysis<ReachingDefs>()));

    const auto Collect = [this](Operation *Candidate) {
      Location Loc = Candidate->getLoc();
      if (const auto &Fused = Loc.dyn_cast<FusedLoc>()) {
        Loc = Fused.getLocations().front();
      }
      if (const auto &Begin = Loc.dyn_cast<FileLineColLoc>()) {
        Ops[Begin.getLine()].push_back(Candidate);
      }
    };

    Function.walk([&Collect](air::StoreOp Store) { Collect(Store); });
    Function.walk([&Collect](air::LoadOp Load) { Collect(Load); });
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpCollector);

private:
  OpsByLineNumber &Ops;
  std::unique_ptr<ReachingDefs> &Analysis;
};

class ReachingDefsTest {
public:
  void run(const Twine &Snippet) {
    IR = frontend::runClangOnCode(Snippet);
    REQUIRE(IR != nullptr);

    MLIRContext &Context = IR->Context;
    Context.disableMultithreading();
    PassManager PM(&Context);

    PM.addNestedPass<FuncOp>(std::make_unique<OpCollector>(Ops, Analysis));

    REQUIRE(succeeded(PM.run(IR->Module)));
  }

  auto getDefinitions(Operation &At, Value For) {
    return Analysis->getDefinitions(At, For);
  }
  auto getDefinitions(LineNumber At, LineNumber For) {
    return getDefinitions(*getOpsByLineNumber(At)[0],
                          getValueByLineNumber(For));
  }

  ReachingDefs::Definitions getDefinitions(LineNumber At) {
    for (auto *Op : getOpsByLineNumber(At))
      if (auto Load = llvm::dyn_cast<air::LoadOp>(Op)) {
        return getDefinitions(*Load, Load.getAddress());
      }
    return {};
  }

  const std::vector<Operation *> &getOpsByLineNumber(LineNumber LN) const {
    const auto It = Ops.find(LN + 1);
    REQUIRE(It != Ops.end());
    REQUIRE(!It->getSecond().empty());
    return It->getSecond();
  }
  Value getValueByLineNumber(LineNumber LN) const {
    for (auto *Op : getOpsByLineNumber(LN))
      if (Op->getOpResults().size() != 0)
        return Op->getOpResult(0);
    FAIL();
    return {};
  }
  Value getDefinitionByLineNumber(LineNumber LN) const {
    for (auto *Op : getOpsByLineNumber(LN))
      if (auto Store = dyn_cast<air::StoreOp>(Op))
        return Store.what();
    FAIL();
    return {};
  }
  ReachingDefs::Definitions
  getExpectedDefinitions(std::initializer_list<LineNumber> LNs) {
    ReachingDefs::Definitions Result;
    llvm::transform(LNs, std::back_inserter(Result), [this](LineNumber LN) {
      return getDefinitionByLineNumber(LN);
    });
    return Result;
  }

private:
  std::unique_ptr<frontend::Output> IR;

protected:
  OpsByLineNumber Ops;
  std::unique_ptr<ReachingDefs> Analysis;
};
} // end anonymous namespace

#define NDEBUG 1

TEST_CASE_METHOD(ReachingDefsTest, "Trivial load-store pair",
                 "[analysis][reaching-defs]") {
  run(R"(
void foo() {
  int x = 1;
  int y = x;
}
)");
  CHECK(getDefinitions(3) == getExpectedDefinitions({2}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Disjoint load-store pairs",
                 "[analysis][reaching-defs]") {
  run(R"(
void foo() {
  int a = 1;
  int b = 2;
  int c = a;
  int d = b;
}
)");
  CHECK(getDefinitions(4) == getExpectedDefinitions({2}));
  CHECK(getDefinitions(5) == getExpectedDefinitions({3}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Trivial redefinition",
                 "[analysis][reaching-defs]") {
  run(R"(
void foo() {
  int x = 1;
  x = 2;
  int y = x;
}
)");
  CHECK(getDefinitions(4) == getExpectedDefinitions({3}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Multiple reaching definitions #1",
                 "[analysis][reaching-defs]") {
  run(R"(
void foo(int x) {
  int a = 1;
  if (x == 42)
    a = 2;
  else
    a = 3;
  int b = a;
}
)");
  CHECK(getDefinitions(7) == getExpectedDefinitions({4, 6}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Multiple reaching definitions #2",
                 "[analysis][reaching-defs]") {
  run(R"(
void foo(int x) {
  int a = 1;
  if (x == 42)
    a = 2;
  int b = a;
}
)");
  CHECK(getDefinitions(5) == getExpectedDefinitions({2, 4}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Multiple reaching definitions #3",
                 "[analysis][reaching-defs]") {
  run(R"(
void foo(int x) {
  int i = 1;
  while (i < x)
    i = i + 1;
  int y = i;
}
)");
  CHECK(getDefinitions(5) == getExpectedDefinitions({2, 4}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Writing to alias #1",
                 "[analysis][reaching-defs]") {
  run(R"(
void foo() {
  int a = 1;
  int *b = &a;
  *b = 2;
  int c = a;
}
)");
  // TODO: improve alias analysis to actually have {4}
  CHECK(getDefinitions(5) == getExpectedDefinitions({}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Writing to alias #2",
                 "[analysis][reaching-defs]") {
  run(R"(
void foo() {
  int a = 1;
  int &b = a;
  b = 2;
  int c = a;
}
)");
  CHECK(getDefinitions(5) == getExpectedDefinitions({4}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Writing to alias #3",
                 "[analysis][reaching-defs]") {
  run(R"(
void foo() {
  int a = 1;
  int *b = &a;
  *b = 2;
  a = 3;
  int c = a;
}
)");
  CHECK(getDefinitions(6) == getExpectedDefinitions({5}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Writing to alias #4",
                 "[analysis][reaching-defs]") {
  run(R"(
void foo() {
  int a = 1;
  int *b = &a;
  int **c = &b;
  int ***d = &c;
  ***d = 2;
  int e = a;
}
)");
  // TODO: improve alias analysis to actually have {6}
  CHECK(getDefinitions(7) == getExpectedDefinitions({}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Possible write on escape",
                 "[analysis][reaching-defs]") {
  run(R"(
void bar(int &);
void foo() {
  int a = 1;
  bar(a);
  int c = a;
}
)");
  CHECK(getDefinitions(5) == getExpectedDefinitions({}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Possible write on alias escape #1",
                 "[analysis][reaching-defs]") {
  run(R"(
void bar(int *);
void foo() {
  int a = 1;
  int *b = &a;
  bar(b);
  int c = a;
}
)");
  CHECK(getDefinitions(6) == getExpectedDefinitions({}));
}

TEST_CASE_METHOD(ReachingDefsTest, "Possible write on alias escape #2",
                 "[analysis][reaching-defs]") {
  run(R"(
void bar(int ***);
void foo() {
  int a = 1;
  int *b = &a;
  int **c = &b;
  bar(&c);
  int d = a;
}
)");
  CHECK(getDefinitions(7) == getExpectedDefinitions({}));
}
