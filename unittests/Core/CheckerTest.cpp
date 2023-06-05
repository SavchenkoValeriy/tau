#include "tau/Core/Checker.h"
#include "tau/AIR/AirOps.h"
#include "tau/Core/CheckerPass.h"
#include "tau/Core/State.h"
#include "tau/Frontend/Clang/Clang.h"
#include "tau/Frontend/Output.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

using namespace tau;
using namespace air;
using namespace core;
using namespace mlir;
using namespace mlir::func;
using namespace llvm;

namespace {

using TrivialState = State<0, 1>;
using MockState = TrivialState;
consteval auto makeTrivialStateMachine() {
  StateMachine<MockState> Result;
  Result.markInitial(TrivialState::getErrorState(0));
  return Result;
}
constexpr auto TrivialStateMachine = makeTrivialStateMachine();

class MockChecker : public CheckerBase<MockChecker, MockState,
                                       TrivialStateMachine, ConstantIntOp> {
public:
  StringRef getName() const override { return "Simple test checker"; }
  StringRef getArgument() const override { return "test-checker"; }
  StringRef getDescription() const override { return "Simple test checker"; }

  MAKE_MOCK1(process, void(ConstantIntOp));

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MockChecker);
};

class CheckerTest {
public:
  void run(const Twine &Snippet) {
    IR = frontend::runClangOnCode(Snippet);
    REQUIRE(IR != nullptr);

    MLIRContext &Context = IR->Context;
    Context.disableMultithreading();
    PassManager PM(&Context);

    SmallVector<AbstractChecker *, 1> Checkers{&TestChecker};

    PM.addNestedPass<FuncOp>(createCheckerPass(Checkers));

    REQUIRE(succeeded(PM.run(IR->Module)));
  }

private:
  std::unique_ptr<frontend::Output> IR;

protected:
  MockChecker TestChecker;
};

inline auto constIntOp(int Expected) {
  return trompeloeil::make_matcher<ConstantIntOp>(

      // predicate lambda that checks the condition
      [Expected](ConstantIntOp Op) {
        return Op.getValue().getExtValue() == Expected;
      },

      // print lambda for error message
      [Expected](std::ostream &OS) {
        OS << " matching constIntOp(" << Expected << ")";
      });
}
} // end anonymous namespace

TEST_CASE_METHOD(CheckerTest, "Checker processing interface",
                 "[analysis][checker]") {
  SECTION("No triggers") {
    FORBID_CALL(TestChecker, process(trompeloeil::_));
    run(R"(
void foo() {
}
)");
  }

  SECTION("Trivial trigger") {
    REQUIRE_CALL(TestChecker, process(constIntOp(1)));
    run(R"(
void foo() {
  1;
}
)");
  }

  SECTION("Multiple triggers") {
    REQUIRE_CALL(TestChecker, process(constIntOp(1))).TIMES(5);
    run(R"(
void foo() {
  1;
  1;
  1;
  1;
  1;
}
)");
  }

  SECTION("Triggers on non-trivial CFG") {
    REQUIRE_CALL(TestChecker, process(constIntOp(1)));
    REQUIRE_CALL(TestChecker, process(constIntOp(2)));
    REQUIRE_CALL(TestChecker, process(constIntOp(3)));
    REQUIRE_CALL(TestChecker, process(constIntOp(4)));
    REQUIRE_CALL(TestChecker, process(constIntOp(5)));
    REQUIRE_CALL(TestChecker, process(constIntOp(6)));
    REQUIRE_CALL(TestChecker, process(constIntOp(7)));
    REQUIRE_CALL(TestChecker, process(constIntOp(8)));
    REQUIRE_CALL(TestChecker, process(constIntOp(9)));
    run(R"(
void foo(bool x, bool y, bool z) {
  1;
  if (x) {
    2;
    while (y) {
      3;
      if (z) {
        4;
      } else {
        5;
      }
      6;
    }
    7;
  } else {
    8;
  }
  9;
}
)");
  }
}

namespace {
class TrivialChecker
    : public CheckerBase<TrivialChecker, TrivialState, TrivialStateMachine> {
public:
  StringRef getName() const override { return "Simple test checker"; }
  StringRef getArgument() const override { return "test-checker"; }
  StringRef getDescription() const override { return "Simple test checker"; }

  mlir::InFlightDiagnostic emitError(mlir::Operation *, TrivialState) {
    return {};
  }
  void emitNote(mlir::InFlightDiagnostic &, mlir::Operation *, TrivialState) {}
};

class CheckerWithOps
    : public CheckerBase<CheckerWithOps, TrivialState, TrivialStateMachine,
                         LoadOp, StoreOp> {
public:
  StringRef getName() const override { return "Simple test checker"; }
  StringRef getArgument() const override { return "test-checker"; }
  StringRef getDescription() const override { return "Simple test checker"; }

  mlir::InFlightDiagnostic emitError(mlir::Operation *, TrivialState) {
    return {};
  }
  void emitNote(mlir::InFlightDiagnostic &, mlir::Operation *, TrivialState) {}

  void process(LoadOp X) const {}
  void process(StoreOp X) const {}
};

class CheckerWithNoSpecialMethods
    : public CheckerBase<CheckerWithNoSpecialMethods, TrivialState,
                         TrivialStateMachine, LoadOp, StoreOp> {
public:
  StringRef getName() const override { return "Simple test checker"; }
  StringRef getArgument() const override { return "test-checker"; }
  StringRef getDescription() const override { return "Simple test checker"; }
};

class CheckerWithNoEmitError
    : public CheckerBase<CheckerWithNoEmitError, TrivialState,
                         TrivialStateMachine> {
public:
  StringRef getName() const override { return "Simple test checker"; }
  StringRef getArgument() const override { return "test-checker"; }
  StringRef getDescription() const override { return "Simple test checker"; }

  void emitNote(mlir::InFlightDiagnostic &, mlir::Operation *, TrivialState) {}
};

class CheckerWithNoEmitNote
    : public CheckerBase<CheckerWithNoEmitNote, TrivialState,
                         TrivialStateMachine> {
public:
  StringRef getName() const override { return "Simple test checker"; }
  StringRef getArgument() const override { return "test-checker"; }
  StringRef getDescription() const override { return "Simple test checker"; }

  mlir::InFlightDiagnostic emitError(mlir::Operation *, TrivialState) {
    return {};
  }
};

class CheckerWithNoProcessMethods
    : public CheckerBase<CheckerWithNoProcessMethods, TrivialState,
                         TrivialStateMachine, LoadOp, StoreOp> {
public:
  StringRef getName() const override { return "Simple test checker"; }
  StringRef getArgument() const override { return "test-checker"; }
  StringRef getDescription() const override { return "Simple test checker"; }

  mlir::InFlightDiagnostic emitError(mlir::Operation *, TrivialState) {
    return {};
  }
  void emitNote(mlir::InFlightDiagnostic &, mlir::Operation *, TrivialState) {}
};

using ComplexState = State<5, 2>;

constexpr auto A = ComplexState::getNonErrorState(0);
constexpr auto B = ComplexState::getNonErrorState(1);
constexpr auto C = ComplexState::getNonErrorState(2);
constexpr auto D = ComplexState::getNonErrorState(3);
constexpr auto E = ComplexState::getNonErrorState(4);

constexpr auto X = ComplexState::getErrorState(0);
constexpr auto Y = ComplexState::getErrorState(1);

template <class Derived, StateMachine<ComplexState> SM>
class StateMachineTestCheckerBase
    : public CheckerBase<Derived, ComplexState, SM> {
public:
  StringRef getName() const override { return "Simple test checker"; }
  StringRef getArgument() const override { return "test-checker"; }
  StringRef getDescription() const override { return "Simple test checker"; }

  mlir::InFlightDiagnostic emitError(mlir::Operation *, ComplexState) {
    return {};
  }
  void emitNote(mlir::InFlightDiagnostic &, mlir::Operation *, ComplexState) {}
};

constexpr StateMachine<ComplexState> EmptyStateMachine;

class EmptyStateMachineChecker
    : public StateMachineTestCheckerBase<EmptyStateMachineChecker,
                                         EmptyStateMachine> {};

consteval auto makeStateMachineWithALoop() {
  StateMachine<ComplexState> Result;
  Result.addEdge(A, B);
  Result.addEdge(B, C);
  Result.addEdge(C, D);
  Result.addEdge(D, E);
  Result.addEdge(E, X);
  Result.addEdge(D, Y);
  Result.addEdge(D, A);
  Result.markInitial(A);
  return Result;
}

constexpr auto LoopedStateMachine = makeStateMachineWithALoop();

class LoopedStateMachineChecker
    : public StateMachineTestCheckerBase<LoopedStateMachineChecker,
                                         LoopedStateMachine> {};

consteval auto makeStateMachineWithUnreachableState() {
  StateMachine<ComplexState> Result;
  Result.addEdge(A, B);
  Result.addEdge(B, C);
  Result.addEdge(C, D);
  Result.addEdge(D, E);
  Result.addEdge(E, X);
  Result.addEdge(D, Y);
  Result.markInitial(B);
  return Result;
}

constexpr auto StateMachineWithUnreachableState =
    makeStateMachineWithUnreachableState();

class StateMachineWithUnreachableStateChecker
    : public StateMachineTestCheckerBase<
          StateMachineWithUnreachableStateChecker,
          StateMachineWithUnreachableState> {};

consteval auto makeStateMachineWithNoPathToError() {
  StateMachine<ComplexState> Result;
  Result.addEdge(A, B);
  Result.addEdge(B, C);
  Result.addEdge(D, E);
  Result.addEdge(E, X);
  Result.addEdge(D, Y);
  Result.markInitial(A);
  Result.markInitial(D);
  return Result;
}

constexpr auto StateMachineWithNoPathToError =
    makeStateMachineWithNoPathToError();

class StateMachineWithNoPathToErrorChecker
    : public StateMachineTestCheckerBase<StateMachineWithNoPathToErrorChecker,
                                         StateMachineWithNoPathToError> {};

consteval auto makeComplexCorrectStateMachine() {
  StateMachine<ComplexState> Result;
  Result.addEdge(A, B);
  Result.addEdge(A, C);
  Result.addEdge(C, D);
  Result.addEdge(D, X);
  Result.addEdge(B, X);
  Result.addEdge(E, B);
  Result.addEdge(B, Y);
  Result.markInitial(A);
  Result.markInitial(E);
  return Result;
}

constexpr auto ComplexCorrectStateMachine = makeComplexCorrectStateMachine();

class ComplexCorrectStateMachineChecker
    : public StateMachineTestCheckerBase<ComplexCorrectStateMachineChecker,
                                         ComplexCorrectStateMachine> {};

} // end anonymous namespace

TEST_CASE("Checker concept validation") {
  STATIC_REQUIRE(Checker<TrivialChecker>);
  STATIC_REQUIRE(Checker<CheckerWithOps>);

  STATIC_REQUIRE_FALSE(Checker<int>);
  STATIC_REQUIRE_FALSE(Checker<CheckerTest>);

  STATIC_REQUIRE_FALSE(Checker<CheckerWithNoSpecialMethods>);
  STATIC_REQUIRE_FALSE(Checker<CheckerWithNoEmitError>);
  STATIC_REQUIRE_FALSE(Checker<CheckerWithNoEmitNote>);
  STATIC_REQUIRE_FALSE(Checker<CheckerWithNoProcessMethods>);

  STATIC_REQUIRE_FALSE(Checker<EmptyStateMachineChecker>);
  STATIC_REQUIRE_FALSE(Checker<LoopedStateMachineChecker>);
  STATIC_REQUIRE_FALSE(Checker<StateMachineWithUnreachableStateChecker>);
  STATIC_REQUIRE_FALSE(Checker<StateMachineWithNoPathToErrorChecker>);
  STATIC_REQUIRE(Checker<ComplexCorrectStateMachineChecker>);
}
