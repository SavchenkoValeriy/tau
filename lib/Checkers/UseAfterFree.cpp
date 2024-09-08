#include "tau/AIR/AirOps.h"
#include "tau/Checkers/Checkers.h"
#include "tau/Core/Checker.h"
#include "tau/Core/CheckerRegistry.h"
#include "tau/Core/State.h"

using namespace tau;
using namespace core;
using namespace air;
using namespace mlir;

using FreeState = core::State<2, 1>;
constexpr FreeState ALLOCATED = FreeState::getNonErrorState(0);
constexpr FreeState FREED = FreeState::getNonErrorState(1);
constexpr FreeState ERROR = FreeState::getErrorState(0);

consteval auto makeStateMachine() {
  StateMachine<FreeState> SM;
  SM.markInitial(ALLOCATED);
  SM.addEdge(ALLOCATED, FREED);
  SM.addEdge(FREED, ERROR);
  return SM;
}

constexpr auto SM = makeStateMachine();

namespace {

class UseAfterFree
    : public CheckerBase<UseAfterFree, FreeState, SM, AllocaOp, HeapAllocaOp,
                         LoadOp, GetFieldPtr, DeallocaOp, HeapDeallocaOp> {
public:
  StringRef getName() const override { return "Use after free checker"; }
  StringRef getArgument() const override { return "use-after-free"; }
  StringRef getDescription() const override {
    return "Detect uses of deallocated values";
  }

  void process(AllocaOp Alloca) const {
    markResultChange<ALLOCATED>(Alloca.getOperation());
  }

  void process(HeapAllocaOp HeapAlloca) const {
    markResultChange<ALLOCATED>(HeapAlloca.getOperation());
  }

  void process(LoadOp Load) const {
    markChange<FREED, ERROR>(Load.getOperation(), Load.getAddress());
  }

  void process(GetFieldPtr Field) const {
    markChange<FREED, ERROR>(Field.getOperation(), Field.getRecord());
  }

  void process(DeallocaOp Dealloca) const {
    markChange<ALLOCATED, FREED>(Dealloca.getOperation(), Dealloca.getPtr());
  }

  void process(HeapDeallocaOp HeapDealloca) const {
    markChange<ALLOCATED, FREED>(HeapDealloca.getOperation(),
                                 HeapDealloca.getPtr());
  }

  InFlightDiagnostic emitError(mlir::Operation *Op, FreeState State) {
    assert(State == ERROR);
    return Op->emitError("Use of deallocated pointer");
  }

  void emitNote(InFlightDiagnostic &Diag, mlir::Operation *Op,
                FreeState State) {
    if (State == FREED) {
      Diag.attachNote(Op->getLoc()) << "Deallocated here";
    } else {
      Diag.attachNote(Op->getLoc()) << "Allocated here";
    }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UseAfterFree);
};

} // end anonymous namespace

void chx::registerUseAfterFreeChecker() { CheckerRegistration<UseAfterFree>(); }
