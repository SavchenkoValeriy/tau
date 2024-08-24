#include "tau/AIR/AirOps.h"
#include "tau/Checkers/Checkers.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerRegistry.h"
#include "tau/Core/State.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

#include <memory>

using namespace tau;
using namespace core;
using namespace air;
using namespace mlir;

using UninitState = core::State<2, 1>;
constexpr UninitState UNINIT = UninitState::getNonErrorState(0);
constexpr UninitState INIT = UninitState::getNonErrorState(1);
constexpr UninitState ERROR = UninitState::getErrorState(0);

consteval auto makeStateMachine() {
  StateMachine<UninitState> SM;
  SM.addEdge(UNINIT, ERROR);
  SM.addEdge(UNINIT, INIT);
  SM.markInitial(UNINIT);
  SM.markInitial(ERROR);
  return SM;
}

constexpr auto SM = makeStateMachine();

namespace {

class UseOfUninit
    : public CheckerBase<UseOfUninit, UninitState, SM, StoreOp, LoadOp, NoOp> {
public:
  StringRef getName() const override {
    return "Use of uninitialized value checker";
  }
  StringRef getArgument() const override { return "use-of-uninit"; }
  StringRef getDescription() const override {
    return "Detect uses of uninitialized values";
  }

  void process(StoreOp Store) const {
    mlir::Operation *Address = Store.getAddress().getDefiningOp();
    if (Store.getValue().getDefiningOp<UndefOp>())
      markResultChange<UNINIT>(Address);
    else
      markChange<UNINIT, INIT>(Store, Store.getAddress());
  }

  void process(LoadOp Load) const {
    markChange<UNINIT, ERROR>(Load.getOperation(), Load.getAddress());
  }

  void process(NoOp Noop) const {
    if (Noop.getValue().getDefiningOp<UndefOp>())
      markChange<ERROR>(Noop.getOperation(), Noop.getValue());
  }

  InFlightDiagnostic emitError(mlir::Operation *Op, UninitState State) {
    assert(State == ERROR);
    return Op->emitError("Use of uninitialized value");
  }

  void emitNote(InFlightDiagnostic &Diag, mlir::Operation *Op,
                UninitState State) {
    assert(State == UNINIT);
    Diag.attachNote(Op->getLoc()) << "Declared without initial value here";
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UseOfUninit);
};

} // end anonymous namespace

void chx::registerUseOfUninitChecker() { CheckerRegistration<UseOfUninit>(); }
