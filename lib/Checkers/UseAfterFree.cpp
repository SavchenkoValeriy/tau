#include "tau/AIR/AirOps.h"
#include "tau/Checkers/Checkers.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerRegistry.h"
#include "tau/Core/State.h"

using namespace tau;
using namespace core;
using namespace air;
using namespace mlir;

using FreeState = core::State<1, 1>;
constexpr FreeState FREED = FreeState::getNonErrorState(0);
constexpr FreeState ERROR = FreeState::getErrorState(0);

consteval auto makeStateMachine() {
  StateMachine<FreeState> SM;
  SM.addEdge(FREED, ERROR);
  SM.markInitial(FREED);
  return SM;
}

constexpr auto SM = makeStateMachine();

namespace {

class UseAfterFree : public CheckerBase<UseAfterFree, FreeState, SM, LoadOp,
                                        GetFieldPtr, DeallocaOp> {
public:
  StringRef getName() const override { return "Use after free checker"; }
  StringRef getArgument() const override { return "use-after-free"; }
  StringRef getDescription() const override {
    return "Detect uses of deallocated values";
  }

  void process(LoadOp Load) const {
    markChange<FREED, ERROR>(Load.getOperation(), Load.getAddress());
  }

  void process(GetFieldPtr Field) const {
    markChange<FREED, ERROR>(Field.getOperation(), Field.getRecord());
  }

  void process(DeallocaOp Dealloca) const {
    markChange<FREED>(Dealloca.getOperation(), Dealloca.getPtr());
  }

  InFlightDiagnostic emitError(mlir::Operation *Op, FreeState State) {
    assert(State == ERROR);
    return Op->emitError("Use of deallocated pointer");
  }

  void emitNote(InFlightDiagnostic &Diag, mlir::Operation *Op,
                FreeState State) {
    assert(State == FREED);
    Diag.attachNote(Op->getLoc()) << "Deallocated here";
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UseAfterFree);
};

} // end anonymous namespace

void chx::registerUseAfterFreeChecker() { CheckerRegistration<UseAfterFree>(); }
