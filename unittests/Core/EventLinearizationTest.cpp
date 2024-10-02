#include "tau/AIR/AirOps.h"
#include "tau/Core/Events.h"
#include "tau/Core/TopoOrderEnumerator.h"
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
using namespace air;
using namespace core;
using namespace llvm;
using namespace mlir;
using namespace mlir::func;

namespace {

using OperationsById = llvm::DenseMap<int, Operation *>;

class OperationCollector
    : public PassWrapper<OperationCollector, OperationPass<FuncOp>> {
public:
  OperationCollector(OperationsById &ToFill,
                     std::unique_ptr<TopoOrderBlockEnumerator> &Enumerator)
      : Operations(ToFill), Enumerator(Enumerator) {}

  StringRef getArgument() const override { return "operation-collector"; }
  StringRef getDescription() const override {
    return "Gathers operations from a function";
  }

  void runOnOperation() override {
    FuncOp Function = getOperation();
    Enumerator = std::make_unique<TopoOrderBlockEnumerator>(
        getAnalysis<TopoOrderBlockEnumerator>());
    Function.walk([this](air::ConstantIntOp ConstInt) {
      int Id = ConstInt.getValue().getExtValue();
      REQUIRE(Operations.count(Id) == 0);
      Operations[Id] = ConstInt.getOperation();
    });
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OperationCollector);

private:
  OperationsById &Operations;
  std::unique_ptr<TopoOrderBlockEnumerator> &Enumerator;
};

class EventHierarchyTest {
public:
  void run(const Twine &Snippet) {
    IR = frontend::runClangOnCode(Snippet);
    REQUIRE(IR != nullptr);

    MLIRContext &Context = IR->Context;
    Context.disableMultithreading();
    PassManager PM(&Context);

    PM.addNestedPass<FuncOp>(
        std::make_unique<OperationCollector>(Operations, Enumerator));

    REQUIRE(succeeded(PM.run(IR->Module)));
  }

  LinearChainOfEvents linearizeChainOfEvents(const AbstractEvent &Event) {
    return Hierarchy.linearizeChainOfEvents(Event, *Enumerator);
  }

protected:
  std::unique_ptr<frontend::Output> IR;
  OperationsById Operations;
  std::unique_ptr<TopoOrderBlockEnumerator> Enumerator;
  EventHierarchy Hierarchy;
};

} // end anonymous namespace

TEST_CASE_METHOD(EventHierarchyTest, "Heterogeneous events",
                 "[analysis][events][linearization]") {
  run(R"(
void test() {
  1; 2;
}
)");

  auto &DataFlowEvt = Hierarchy.addDataFlowEvent(Operations[1]);
  auto &StateEvt = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(1)},
                                           Operations[2], {&DataFlowEvt});

  auto Results = linearizeChainOfEvents(&StateEvt);
  REQUIRE(Results.size() == 2);
  CHECK(Results[0].get<const StateEvent *>() == &StateEvt);
  CHECK(Results[1].get<const DataFlowEvent *>() == &DataFlowEvt);
}

TEST_CASE_METHOD(EventHierarchyTest, "Events in various basic blocks",
                 "[analysis][events][linearization]") {
  run(R"(
void test(bool cond) {
  1;
  if (cond) {
    2;
  } else {
    3;
  }
}
)");

  auto &Evt1 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(1)},
                                       Operations[1]);
  auto &Evt2 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(2)},
                                       Operations[2], {&Evt1});
  auto &Evt3 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(3)},
                                       Operations[3], {&Evt2});

  auto Results = linearizeChainOfEvents(&Evt3);
  REQUIRE(Results.size() == 3);
  CHECK(Results[0].get<const StateEvent *>() == &Evt3);
  CHECK(Results[1].get<const StateEvent *>() == &Evt2);
  CHECK(Results[2].get<const StateEvent *>() == &Evt1);
}

TEST_CASE_METHOD(EventHierarchyTest, "Events in one block",
                 "[analysis][events][linearization]") {
  run(R"(
void test() {
  1; 2; 3;
}
)");

  auto &Evt1 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(1)},
                                       Operations[1]);
  auto &Evt2 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(2)},
                                       Operations[2], {&Evt1});
  auto &Evt3 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(3)},
                                       Operations[3], {&Evt2});

  auto Results = linearizeChainOfEvents(&Evt3);
  REQUIRE(Results.size() == 3);
  CHECK(Results[0].get<const StateEvent *>() == &Evt3);
  CHECK(Results[1].get<const StateEvent *>() == &Evt2);
  CHECK(Results[2].get<const StateEvent *>() == &Evt1);
}

TEST_CASE_METHOD(EventHierarchyTest, "Events hierarchy with multiple branches",
                 "[analysis][events][linearization]") {
  run(R"(
void test(bool cond1, bool cond2) {
  1;
  if (cond1) {
    2;
  } else if (cond2) {
    3;
  } else {
    4;
  }
}
)");

  auto &Evt1 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(1)},
                                       Operations[1]);
  auto &Evt2 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(2)},
                                       Operations[2], {&Evt1});
  auto &Evt3 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(3)},
                                       Operations[3], {&Evt1});
  auto &Evt4 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(4)},
                                       Operations[4], {&Evt2, &Evt3});

  auto Results = linearizeChainOfEvents(&Evt4);
  REQUIRE(Results.size() == 4);
  CHECK(Results[0].get<const StateEvent *>() == &Evt4);
  CHECK(Results[1].get<const StateEvent *>() == &Evt3);
  CHECK(Results[2].get<const StateEvent *>() == &Evt2);
  CHECK(Results[3].get<const StateEvent *>() == &Evt1);
}

TEST_CASE_METHOD(EventHierarchyTest, "Events hierarchy with repeated parent",
                 "[analysis][events][linearization]") {
  run(R"(
void test(bool cond) {
  1;
  if (cond) {
    2;
  }
  3;
}
)");

  auto &Evt1 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(1)},
                                       Operations[1]);
  auto &Evt2 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(2)},
                                       Operations[2], {&Evt1});
  auto &Evt3 = Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(3)},
                                       Operations[3], {&Evt1, &Evt2});

  auto Results = linearizeChainOfEvents(&Evt3);
  REQUIRE(Results.size() == 3);
  CHECK(Results[0].get<const StateEvent *>() == &Evt3);
  CHECK(Results[1].get<const StateEvent *>() == &Evt2);
  CHECK(Results[2].get<const StateEvent *>() == &Evt1);
}

TEST_CASE_METHOD(EventHierarchyTest, "Complex combined case",
                 "[analysis][events][linearization]") {
  run(R"(
void test(bool cond1, bool cond2) {
  1;
  if (cond1) {
    2;
    if (cond2) {
      3;
    } else {
      4;
    }
  } else {
    5;
  }
  6;
}
)");

  auto &StateEvt1 = Hierarchy.addStateEvent(
      {"TestChecker", StateID::fromRaw(1)}, Operations[1]);
  auto &DataFlowEvt1 = Hierarchy.addDataFlowEvent(Operations[2], {&StateEvt1});
  auto &StateEvt2 = Hierarchy.addStateEvent(
      {"TestChecker", StateID::fromRaw(2)}, Operations[3], {&DataFlowEvt1});
  auto &StateEvt3 =
      Hierarchy.addStateEvent({"TestChecker", StateID::fromRaw(3)},
                              Operations[4], {&StateEvt1, &StateEvt2});
  auto &DataFlowEvt2 =
      Hierarchy.addDataFlowEvent(Operations[6], {&StateEvt2, &StateEvt3});

  auto Results = linearizeChainOfEvents(&DataFlowEvt2);
  REQUIRE(Results.size() == 5);
  CHECK(Results[0].get<const DataFlowEvent *>() == &DataFlowEvt2);
  CHECK(Results[1].get<const StateEvent *>() == &StateEvt3);
  CHECK(Results[2].get<const StateEvent *>() == &StateEvt2);
  CHECK(Results[3].get<const DataFlowEvent *>() == &DataFlowEvt1);
  CHECK(Results[4].get<const StateEvent *>() == &StateEvt1);
}
