#include "tau/Core/MemoryStore.h"
#include "tau/AIR/AirOps.h"
#include "tau/Core/DataFlowEvent.h"
#include "tau/Core/FlowWorklist.h"
#include "tau/Core/TopoOrderEnumerator.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <iterator>
#include <memory>
#include <string>

using namespace tau;
using namespace core;
using namespace llvm;
using namespace mlir;
using namespace mlir::func;

using Catch::Matchers::UnorderedRangeEquals;

namespace {
using LineNumber = unsigned;
using LoadsByLineNumber = llvm::DenseMap<LineNumber, mlir::Value>;
using MemoryStoresByLineNumber = llvm::DenseMap<LineNumber, MemoryStore>;

class DummyInterpreter
    : public PassWrapper<DummyInterpreter, OperationPass<FuncOp>> {
public:
  DummyInterpreter(LoadsByLineNumber &Loads, MemoryStoresByLineNumber &Stores,
                   DataFlowEventForest &Forest)
      : Loads(Loads), MemoryStores(Stores), Forest(Forest) {}

  StringRef getArgument() const override { return "dummy-interpreter"; }
  StringRef getDescription() const override { return "Gathers memory stores"; }

  void runOnOperation() override {
    FuncOp Function = getOperation();
    if (Function.isDeclaration())
      return;

    const auto Collect = [this](Value Candidate) {
      Location Loc = Candidate.getLoc();
      if (const auto &Fused = Loc.dyn_cast<FusedLoc>()) {
        Loc = Fused.getLocations().front();
      }
      if (const auto &Begin = Loc.dyn_cast<FileLineColLoc>()) {
        Loads[Begin.getLine()] = Candidate;
      }
    };

    Function.walk([&Collect](air::LoadOp Load) { Collect(Load); });

    auto &Worklist = getAnalysis<ForwardWorklist>();
    auto &Enumerator = getAnalysis<TopoOrderBlockEnumerator>();
    BitVector Processed{static_cast<unsigned>(Function.getBlocks().size())};
    Worklist.enqueue(&Function.getBlocks().front());
    SmallVector<MemoryStore, 20> BlockStores(Function.getBlocks().size(),
                                             MemoryStore{Forest});

    const auto SetStore = [&BlockStores](MemoryStore New, unsigned Index) {
      if (BlockStores[Index] == New)
        return false;
      BlockStores[Index] = New;
      return true;
    };

    while (Block *BB = Worklist.dequeue()) {
      MemoryStore CurrentStore{Forest};

      for (Block *Pred : BB->getPredecessors()) {
        CurrentStore =
            CurrentStore.join(BlockStores[Enumerator.getTopoOrderIndex(Pred)]);
      }

      for (Operation &Op : *BB) {
        CurrentStore = CurrentStore.interpret(&Op);
        Location Loc = Op.getLoc();
        if (const auto &Fused = Loc.dyn_cast<FusedLoc>()) {
          Loc = Fused.getLocations().front();
        }
        if (const auto &Begin = Loc.dyn_cast<FileLineColLoc>()) {
          if (const auto Result =
                  MemoryStores.insert({Begin.getLine(), CurrentStore});
              !Result.second) {
            Result.first->getSecond() = CurrentStore;
          }
        }

        const auto Index = Enumerator.getTopoOrderIndex(BB);
        if (SetStore(CurrentStore, Index) || Processed[Index]) {
          Processed[Index] = true;
          Worklist.enqueueSuccessors(*BB);
        }
      }
    }
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DummyInterpreter);

private:
  LoadsByLineNumber &Loads;
  MemoryStoresByLineNumber &MemoryStores;
  DataFlowEventForest &Forest;
};

class MemoryStoreTest {
public:
  void run(const Twine &Snippet, const Twine &Filename = "input.cc") {
    IR = frontend::runClangOnCode(Snippet);
    REQUIRE(IR != nullptr);

    MLIRContext &Context = IR->Context;
    Context.disableMultithreading();
    PassManager PM(&Context);

    PM.addNestedPass<FuncOp>(
        std::make_unique<DummyInterpreter>(Loads, MemoryStores, Forest));

    REQUIRE(succeeded(PM.run(IR->Module)));
  }

  MemoryStore getMemoryStoreByLine(LineNumber LN) const {
    const auto It = MemoryStores.find(LN + 1);
    REQUIRE(It != MemoryStores.end());
    return It->second;
  }

  mlir::Value getLoadByLine(LineNumber LN) const {
    const auto It = Loads.find(LN + 1);
    REQUIRE(It != Loads.end());
    return It->second;
  }

  static unsigned getAsConstant(mlir::Value Value) {
    const auto ToString = [](mlir::Value V) {
      std::string Buffer;
      llvm::raw_string_ostream SS(Buffer);
      V.print(SS);
      return SS.str();
    };
    auto AsConst = llvm::dyn_cast<air::ConstantIntOp>(Value.getDefiningOp());
    INFO("Inspected value is " << ToString(Value));
    REQUIRE(AsConst);
    return AsConst.getValue().getExtValue();
  }

private:
  std::unique_ptr<frontend::Output> IR;

protected:
  LoadsByLineNumber Loads;
  MemoryStoresByLineNumber MemoryStores;
  DataFlowEventForest Forest;
};

} // end anonymous namespace

TEST_CASE_METHOD(MemoryStoreTest, "Trivial load/store",
                 "[analysis][memory-store]") {
  run(R"(
void foo() {
  int *a = nullptr;
  int b = 42;
  a = &b;
  int c = *a;
}
)");

  const auto MS = getMemoryStoreByLine(5);
  auto Load = getLoadByLine(5);

  const auto Set = MS.getDefininingValues(Load);
  CHECK(Set.size() == 1);
  const auto Def = *Set.begin();
  CHECK(getAsConstant(Def.Value) == 42);
}

TEST_CASE_METHOD(MemoryStoreTest, "Field load/store",
                 "[analysis][memory-store]") {
  run(R"(
struct A {
  int *x;
};
void foo() {
  int c = 42;
  A a;
  a.x = &c;
  A b;
  b.x = &c;
  *a.x = 10;
  int d = *b.x;
}
)");

  const auto MS = getMemoryStoreByLine(11);
  auto Load = getLoadByLine(11);

  const auto Set = MS.getDefininingValues(Load);
  CHECK(Set.size() == 1);
  const auto Def = *Set.begin();
  CHECK(getAsConstant(Def.Value) == 10);
}

TEST_CASE_METHOD(MemoryStoreTest, "Join after if-else",
                 "[analysis][memory-store]") {
  run(R"(
void foo(bool cond) {
  int a = 10;
  int b = 20;
  int *c = nullptr;
  if (cond) {
    c = &a;
  } else {
    c = &b;
  }
  int d = *c;
}
)");

  const auto MS = getMemoryStoreByLine(10);
  auto Load = getLoadByLine(10);

  const auto Set = MS.getDefininingValues(Load);
  CHECK(Set.size() == 2);
  std::vector<unsigned> Consts;
  llvm::transform(Set, std::back_inserter(Consts),
                  [this](const MemoryStore::Definition &Def) {
                    return getAsConstant(Def.Value);
                  });

  const std::array<unsigned, 2> Expected{10, 20};
  CHECK_THAT(Consts, UnorderedRangeEquals(Expected));
}

TEST_CASE_METHOD(MemoryStoreTest, "Join after if-else rewrite #1",
                 "[analysis][memory-store]") {
  run(R"(
void foo(bool cond) {
  int a = 10;
  int b = 20;
  int *c = nullptr;
  if (cond) {
    c = &a;
  } else {
    c = &b;
  }
  int d = *c;
  c = &a;
  int e = *c;
}
)");
  {
    const auto MS = getMemoryStoreByLine(10);
    auto Load = getLoadByLine(10);

    const auto Set = MS.getDefininingValues(Load);
    CHECK(Set.size() == 2);
    std::vector<unsigned> Consts;
    llvm::transform(Set, std::back_inserter(Consts),
                    [this](const MemoryStore::Definition &Def) {
                      return getAsConstant(Def.Value);
                    });

    const std::array<unsigned, 2> Expected{10, 20};
    CHECK_THAT(Consts, UnorderedRangeEquals(Expected));
  }
  {
    const auto MS = getMemoryStoreByLine(12);
    auto Load = getLoadByLine(12);

    const auto Set = MS.getDefininingValues(Load);
    CHECK(Set.size() == 1);
    const auto Def = *Set.begin();
    CHECK(getAsConstant(Def.Value) == 10);
  }
}

TEST_CASE_METHOD(MemoryStoreTest, "Join after if-else rewrite #2",
                 "[analysis][memory-store]") {
  run(R"(
void foo(bool cond) {
  int a = 10;
  int b = 20;
  int *c = nullptr;
  if (cond) {
    c = &a;
  } else {
    c = &b;
  }
  int d = *c;
  a = 30;
  int e = *c;
}
)");
  {
    const auto MS = getMemoryStoreByLine(10);
    auto Load = getLoadByLine(10);

    const auto Set = MS.getDefininingValues(Load);
    CHECK(Set.size() == 2);
    std::vector<unsigned> Consts;
    llvm::transform(Set, std::back_inserter(Consts),
                    [this](const MemoryStore::Definition &Def) {
                      return getAsConstant(Def.Value);
                    });

    const std::array<unsigned, 2> Expected{10, 20};
    CHECK_THAT(Consts, UnorderedRangeEquals(Expected));
  }
  {
    const auto MS = getMemoryStoreByLine(12);
    auto Load = getLoadByLine(12);

    const auto Set = MS.getDefininingValues(Load);
    CHECK(Set.size() == 2);
    std::vector<unsigned> Consts;
    llvm::transform(Set, std::back_inserter(Consts),
                    [this](const MemoryStore::Definition &Def) {
                      return getAsConstant(Def.Value);
                    });

    const std::array<unsigned, 2> Expected{20, 30};
    CHECK_THAT(Consts, UnorderedRangeEquals(Expected));
  }
}

TEST_CASE_METHOD(MemoryStoreTest, "Join after if-no-else",
                 "[analysis][memory-store]") {
  run(R"(
void foo(bool cond) {
  int a = 10;
  int b = 20;
  int *c = &a;
  if (cond) {
    c = &b;
  }
  int d = *c;
}
)");

  const auto MS = getMemoryStoreByLine(8);
  auto Load = getLoadByLine(8);

  const auto Set = MS.getDefininingValues(Load);
  CHECK(Set.size() == 2);
  std::vector<unsigned> Consts;
  llvm::transform(Set, std::back_inserter(Consts),
                  [this](const MemoryStore::Definition &Def) {
                    return getAsConstant(Def.Value);
                  });

  const std::array<unsigned, 2> Expected{10, 20};
  CHECK_THAT(Consts, UnorderedRangeEquals(Expected));
}

TEST_CASE_METHOD(MemoryStoreTest, "Deeply nested field load/store",
                 "[analysis][memory-store]") {
  run(R"(
struct A {
  A *a;
  int *b;
};
void foo(A *a) {
  int x = 10;
  a->a->a->a->a->a->a->b = &x;
  x = 20;
  int &c = *a->a->a->a->a->a->a->b;
  int d = c;
}
)");

  const auto MS = getMemoryStoreByLine(10);
  auto Load = getLoadByLine(10);

  const auto Set = MS.getDefininingValues(Load);
  CHECK(Set.size() == 1);
  const auto Def = *Set.begin();
  CHECK(getAsConstant(Def.Value) == 20);
}

TEST_CASE_METHOD(MemoryStoreTest, "Nested rewrite",
                 "[analysis][memory-store]") {
  run(R"(
struct A {
  A *a;
  int *b;
};
void foo(A *a, A *b) {
  int x = 10;
  int y = 20;
  a->a->a->b = &y;
  b->a->b = &x;
  a->a = b;
  x = 40;
  int &c = *a->a->a->b;
  int d = c;
}
)");

  const auto MS = getMemoryStoreByLine(13);
  auto Load = getLoadByLine(13);

  const auto Set = MS.getDefininingValues(Load);
  CHECK(Set.size() == 1);
  const auto Def = *Set.begin();
  CHECK(getAsConstant(Def.Value) == 40);
}
